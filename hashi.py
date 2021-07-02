from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Tuple, Union
import sys


def parse_file(filename: str) -> List[List[int]]:
    """Read a Hashi puzzle from a file."""
    grid = []
    with open(filename) as f:
        for line in f:
            line = line.strip("\n")
            if len(line) == 0:
                break
            row = []
            for c in line:
                if c == " ":
                    row.append(0)
                else:
                    row.append(int(c))

            grid.append(row)

    return grid


@dataclass(eq=True, unsafe_hash=True)
class Node:
    row: int = field(hash=False)
    col: int = field(hash=False)
    target: int
    ix: int


@dataclass(eq=True, unsafe_hash=True)
class Edge:
    a: Node
    b: Node
    val: int = field(hash=False)
    ix: int


@dataclass(frozen=True)
class Action:
    ix: int
    num: int


def intersects(e: Edge, f: Edge) -> bool:
    # figure out which is horizontal
    if e.a.row == e.b.row and e.a.col != e.b.col and \
       f.a.row != f.b.row and f.a.col == f.b.col:
        # e is horizontal, f is vertical
        horiz = e
        vert = f
    elif e.a.row != e.b.row and e.a.col == e.b.col and \
           f.a.row == f.b.row and f.a.col != f.b.col:
        # e is vertical, f is horizontal
        vert = e
        horiz = f
    else:
        raise ValueError(f"intersects needs one horizontal edge, \
                          one vertical, got {e}, {f}")

    i = horiz.a.row
    j1 = min(horiz.a.col, horiz.b.col)
    j2 = max(horiz.a.col, horiz.b.col)

    j = vert.a.col
    i1 = min(vert.a.row, vert.b.row)
    i2 = max(vert.a.row, vert.b.row)

    return i1 < i < i2 and j1 < j < j2


MAX_VAL = 2


class Board:
    def __init__(self, grid: List[List[int]]):
        self.height = len(grid)
        self.width = len(grid[0]) if self.height > 0 else 0

        node_map = {}
        nodes = []
        k = 0
        for i in range(self.height):
            assert len(grid[i]) == self.width, \
                f"line {j}: expected {self.width} chars, got {len(grid[i])}"
            for j in range(self.width):
                target = grid[i][j]
                if target != 0:
                    n = Node(i, j, target, k)
                    node_map[(i, j)] = n
                    nodes.append(n)
                    k += 1

        # horizontal edges
        neighbors = defaultdict(list)
        edges = []
        horizontal_edges = []
        for i in range(self.height):
            # find first nonzero
            for j in range(self.width):
                if grid[i][j] != 0:
                    prev = j
                    break
            else:
                # didn't find an edge, continue
                continue

            done = False
            while not done:
                for j in range(prev + 1, self.width):
                    if grid[i][j] != 0:
                        a = node_map[(i, prev)]
                        b = node_map[(i, j)]

                        e = Edge(a, b, 0, len(edges))
                        neighbors[a].append(e)
                        neighbors[b].append(e)
                        horizontal_edges.append(e)
                        edges.append(e)

                        prev = j
                        break
                else:
                    done = True

        # vertical_edges
        collisions = defaultdict(list)
        for j in range(self.width):
            # find first nonempty
            for i in range(self.height):
                if grid[i][j] != 0:
                    prev = i
                    break
            else:
                # didn't find a nonempty, continue
                continue

            done = False
            while not done:
                for i in range(prev + 1, self.height):
                    if grid[i][j] != 0:
                        a = node_map[(prev, j)]
                        b = node_map[(i, j)]

                        e = Edge(a, b, 0, len(edges))
                        neighbors[a].append(e)
                        neighbors[b].append(e)
                        edges.append(e)

                        # check for collisions with horizontal edges
                        for h in horizontal_edges:
                            if intersects(e, h):
                                collisions[e].append(h)
                                collisions[h].append(e)

                        prev = i
                        break
                else:
                    done = True

        self.neighbors: Dict[Node, List[Edge]] = neighbors
        self.edges: List[Edge] = edges
        self.nodes: List[Node] = nodes

        self.collisions: Dict[Edge, List[Edge]] = collisions

    def edges_taken(self, node: Node) -> int:
        """Get the number of edges in contact with a node."""
        return sum(e.val for e in self.neighbors[node])

    def edges_left(self, node: Node) -> int:
        """Get the number of edges a node still needs to meet its target."""
        return node.target - self.edges_taken(node)

    def blocked_by(self, edge: Edge) -> List[Edge]:
        """Get the edges that are blocking an edge."""
        return [e for e in self.collisions[edge] if e.val > 0]

    def do(self, move: List[Action]):
        """Perform a list of actions, adding it to the backtrack stack."""
        for action in move:
            self._do_action(action)

    def _do_action(self, action: Action):
        e = self.edges[action.ix]
        num = action.num

        # assert edge has enough space left
        assert e.val + num <= MAX_VAL, \
            f"Action {action} puts {e} over allowed edges"

        # assert nodes have enough space left
        assert self.edges_left(e.a) >= num, \
            f"Action {action} puts {e.a} over target edges"
        assert self.edges_left(e.b) >= num, \
            f"Action {action} puts {e.b} over target edges"

        # assert no conflicting edges are filled
        bb = self.blocked_by(e)
        assert len(bb) == 0, \
            f"Action {action} adds to edge with blocking edges {bb}"

        e.val += num

    def undo(self, move: List[Action]):
        """Pop the last action from the backtrack stack and undo it."""
        for action in move:
            self._undo_action(action)

    def _undo_action(self, action: Action):
        e = self.edges[action.ix]
        num = action.num

        # Check edge has enough edges to remove
        assert e.val >= num, f"Tried to remove {num} edges from {e}"

        # Check each node has enough space to remove
        assert self.edges_taken(e.a) >= num, \
            f"Can't remove {num} edges from {e.a}"
        assert self.edges_taken(e.b) >= num, \
            f"Can't remove {num} edges from {e.b}"

        e.val -= num

    def unblocked_edges(self, node: Node) -> List[Edge]:
        """Get edges adjacent to a node that aren't blocked by another."""
        return [e for e in self.neighbors[node]
                if len(self.blocked_by(e)) == 0]

    def edge_capacity_left(self, edge: Edge) -> int:
        """Get space left on edge."""
        return min(MAX_VAL - edge.val,
                   self.edges_left(edge.a),
                   self.edges_left(edge.b))

    def render(self, out=sys.stdout) -> str:
        """Print the solution out to some stream."""
        def mk_grid(h, w):
            g = []
            for _ in range(h):
                row = []
                for _ in range(w):
                    row.append(" ")
                g.append(row)
            return g

        output = mk_grid(self.height, self.width)
        for n in self.nodes:
            output[n.row][n.col] = str(n.target)

        for e in self.edges:
            if e.a.row == e.b.row:
                if e.val == 0:
                    continue
                ch = "-" if e.val == 1 else "="

                for j in range(e.a.col + 1, e.b.col):
                    output[e.a.row][j] = ch
            else:
                if e.val == 0:
                    continue
                ch = "|" if e.val == 1 else "ǁ"
                for i in range(e.a.row + 1, e.b.row):
                    output[i][e.a.col] = ch

        print("\n".join("".join(row) for row in output), file=out)


def solved(board: Board) -> bool:
    """Determine whether all nodes are satisfied."""
    return all(board.edges_left(node) == 0 for node in board.nodes)


def in_bad_state(board: Board) -> bool:
    """Determine if the board is unsolvable in its current state."""
    for node in board.nodes:
        edges_left = board.edges_left(node)
        unblocked = board.unblocked_edges(node)
        capacity = sum(MAX_VAL - e.val for e in unblocked)
        if capacity < edges_left:
            return True

    return False


@dataclass
class SearchState:
    ix: int
    definite: bool


@dataclass
class BacktrackFrame:
    move: List[Action]
    search_state: SearchState


def find_move(board: Board, search_state: SearchState = SearchState(0, True)) \
        -> Union[BacktrackFrame, None]:
    """Find a move, preferring one that is logically required by the board
    state."""
    # look for an action that must be performed
    if search_state.definite:
        for ix in range(search_state.ix, len(board.nodes)):
            # an adjacent edge needs to be filled if removing it would make it
            # impossible to satisfy the node's target value.
            node = board.nodes[ix]

            edges_left = board.edges_left(node)
            if edges_left == 0:
                continue

            unblocked = board.unblocked_edges(node)
            capacity = sum(board.edge_capacity_left(e) for e in unblocked)
            assert capacity >= edges_left, \
                f"Node {node}'s neighbors {unblocked} don't have enough capacity"

            actions = []
            for e in unblocked:
                capacity_without_e = capacity - board.edge_capacity_left(e)
                if edges_left > capacity_without_e:
                    num = min(MAX_VAL, edges_left - capacity_without_e)
                    actions.append(Action(e.ix, num))

            if len(actions) > 0:
                return BacktrackFrame(actions,
                                      SearchState(ix, True))

    # naively iterate over edges
    start_ix = 0 if search_state.definite else search_state.ix
    for ix in range(start_ix, len(board.edges)):
        edge = board.edges[ix]

        if len(board.blocked_by(edge)) > 0:
            # blocked; continue
            continue

        if board.edge_capacity_left(edge) > 0 and \
            board.edges_left(edge.a) > 0 and \
            board.edges_left(edge.b) > 0:
            return BacktrackFrame([Action(edge.ix, 1)],
                                  SearchState(edge.ix, False))

    return None


def backtrack(board: Board, backtrack_stack: List[BacktrackFrame]) -> SearchState:
    """Go back up the stack undoing actions until we find a non-definite
    action."""
    assert len(backtrack_stack) > 0, "Backtrack stack is empty"
    frame = backtrack_stack.pop()
    board.undo(frame.move)

    while frame.search_state.definite:
        frame = backtrack_stack.pop()
        board.undo(frame.move)

    if frame.search_state.definite:
        # all actions definite and still failed, unsolvable puzzle
        raise ValueError("This puzzle isn't solvable")

    return frame


def solve(board: Board, maxiter=10000):
    backtrack_stack: List[BacktrackFrame] = []
    iters = 0

    while iters < maxiter and not solved(board):
        if in_bad_state(board):
            start_at = backtrack(board, backtrack_stack)
            ss = find_move(board, start_at)
        else:
            ss = find_move(board)
            if ss is None:
                ss = backtrack(board, backtrack_stack)
                ss.ix += 1
                ss = find_move(board, ss)

        if ss is None:
            raise Exception("Puzzle unsolvable or rules incomplete")

        board.do(ss.move)
        backtrack_stack.append(ss)

        iters += 1


if __name__ == "__main__":
    args = sys.argv
    file = args[1]

    grid = parse_file(file)
    board = Board(grid)

    solve(board)
    board.render()
