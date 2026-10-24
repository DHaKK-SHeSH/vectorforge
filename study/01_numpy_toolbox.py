"""Study 01: the NumPy moves that every VectorForge index uses.

Run it from the project root:
    python study/01_numpy_toolbox.py

It shows ONE idea at a time and waits for Enter before the next one.
Every variable is printed with its shape. Read the code of each section
next to its output. Change the numbers and run it again: that's the best
way to make it stick.
"""

import numpy as np

np.set_printoptions(suppress=True)


def show(name, value):
    value = np.asarray(value)
    print(f"\n  {name}   shape {value.shape}")
    for line in str(value).splitlines():
        print("      " + line)


def section(n, title, idea):
    print("\n" + "=" * 70)
    print(f" {n}. {title}")
    print("=" * 70)
    print(" " + idea)


def pause():
    try:
        input("\n[Enter] for the next section ... ")
    except EOFError:
        pass


# ---------------------------------------------------------------------------
section(1, "Shapes: a vector is a row, many vectors are a table",
        "Every index stores vectors as a table: one ROW per vector.\n"
        " shape (n, d) = n vectors, d numbers each.")
x = np.array([[1, 2],
              [3, 4],
              [5, 6]], dtype=np.float32)
show("x  (3 vectors, 2 numbers each)", x)
show("x[0]  (row 0 = vector 0)", x[0])
print("\n  len(x) =", len(x), "   <- number of vectors (rows)")
pause()

# ---------------------------------------------------------------------------
section(2, "Picking rows and columns:  ':' means 'all'",
        "x[i] = row i.   x[:, j] = column j of EVERY row.   The comma separates rows | columns.")
show("x[1]      row 1", x[1])
show("x[:, 0]   column 0 of every row", x[:, 0])
show("x[:, 1]   column 1 of every row", x[:, 1])
show("x[1:3]    rows 1 and 2 (slice: from 1 up to, not including, 3)", x[1:3])
print("\n  Careful:  x[:1] (rows up to 1)  is NOT  x[:, 1] (column 1).  The comma matters.")
pause()

# ---------------------------------------------------------------------------
section(3, "Summing along an axis",
        "axis=1 adds ACROSS each row (one result per row).\n"
        " axis=0 adds DOWN each column (one result per column).")
show("x ** 2", x ** 2)
show("(x ** 2).sum(axis=1)   one number per vector = its squared length", (x ** 2).sum(axis=1))
show("(x ** 2).sum(axis=0)   one number per column", (x ** 2).sum(axis=0))
print("\n  flat.py, assign() and PQ all use .sum(axis=1): 'squared length of every vector'.")
pause()

# ---------------------------------------------------------------------------
section(4, "[:, None]: turn a row of numbers into a column, to build a table",
        "a (2,) + b (3,) doesn't work.  a[:, None] (2,1) + b (3,) gives a (2, 3) TABLE:\n"
        " every a combined with every b.  This is how the distance matrix is built.")
a = np.array([10, 20])
b = np.array([1, 2, 3])
show("a", a)
show("a[:, None]   same numbers, now a column", a[:, None])
show("b", b)
show("a[:, None] + b   row i, column j = a[i] + b[j]", a[:, None] + b)
print("\n  In flat.py:  q_norms[:, None] - 2 * (xq @ xb.T) + xb_norms")
print("  rows = queries, columns = database vectors.  Same trick.")
pause()

# ---------------------------------------------------------------------------
section(5, "The distance table with @",
        "xq @ xb.T = every query's dot product with every vector: shape (nq, n).\n"
        " With the norms from section 3 and the [:, None] from section 4,\n"
        " that's ALL squared distances in one line.")
xq = np.array([[0, 0],
               [5, 5]], dtype=np.float32)          # 2 queries
xb = x                                              # 3 vectors from section 1
show("xq  (2 queries)", xq)
show("xb  (3 vectors)", xb)
show("xq @ xb.T   (2, 3): query i . vector j", xq @ xb.T)
q_norms = (xq ** 2).sum(axis=1)
xb_norms = (xb ** 2).sum(axis=1)
D = q_norms[:, None] - 2 * (xq @ xb.T) + xb_norms
show("D = q_norms[:, None] - 2 * (xq @ xb.T) + xb_norms", D)
print("\n  Check row 0, column 0:  query (0,0) to vector (1,2):  1^2 + 2^2 = 5  (correct)")
print("  D[i, j] = squared distance from query i to vector j.  Rows = queries.")
pause()

# ---------------------------------------------------------------------------
section(6, "Indexing with a LIST of numbers: pick entries in that order",
        "arr[[2, 0, 2]] = entries 2, 0, 2.   Used by decode, PQ search, IVF.")
cat = np.array([[10, 10],
                [20, 20],
                [30, 30]])
show("cat  (a catalogue with 3 entries)", cat)
show("cat[[2, 0, 2]]   entries 2, 0, 2", cat[[2, 0, 2]])
table = np.array([[724, 164, 4],
                  [ 50,  60, 70]])
show("table  (2 queries x 3 entries)", table)
show("table[:, [2, 0, 2]]   every row (:), columns 2, 0, 2", table[:, [2, 0, 2]])
print("\n  PQ search:  table[:, codes[:, j]]  = 'for every query, the entries these codes point to'.")
pause()

# ---------------------------------------------------------------------------
section(7, "True/False masks and np.flatnonzero",
        "labels == 0 gives True/False for every position.\n"
        " np.flatnonzero(...) gives the POSITIONS that are True.  Used by IVF add.")
labels = np.array([0, 0, 1, 0, 1])
show("labels   (cell of each of 5 vectors)", labels)
show("labels == 0", labels == 0)
show("np.flatnonzero(labels == 0)   ids in cell 0", np.flatnonzero(labels == 0))
show("np.flatnonzero(labels == 1)   ids in cell 1", np.flatnonzero(labels == 1))
lists = [np.flatnonzero(labels == c) for c in range(2)]
print("\n  lists = [np.flatnonzero(labels == c) for c in range(2)]")
for c, ids in enumerate(lists):
    print(f"      lists[{c}] = {ids}   ({len(ids)} ids)")
print("\n  A LIST of arrays, because cells have different sizes (3 and 2 here).")
pause()

# ---------------------------------------------------------------------------
section(8, "reshape: cut every vector into pieces (PQ)",
        "(n, d) -> (n, m, dsub).  The numbers don't change, only the grouping.\n"
        " pieces[:, j, :] = piece j of EVERY vector.")
v = np.array([[1, 1, 9, 9],
              [2, 2, 8, 8],
              [3, 3, 7, 7]])
show("v  (3 vectors of 4 numbers)", v)
pieces = v.reshape(3, 2, 2)
show("pieces = v.reshape(3, 2, 2)   (3 vectors, 2 pieces, 2 numbers)", pieces)
show("pieces[0]        vector 0 cut in 2", pieces[0])
show("pieces[:, 0, :]  piece 0 of every vector", pieces[:, 0, :])
show("pieces[:, 1, :]  piece 1 of every vector", pieces[:, 1, :])
pause()

# ---------------------------------------------------------------------------
section(9, "Putting arrays together: stack vs concatenate",
        "np.stack  = pile SAME-shaped arrays along a NEW axis.\n"
        " np.concatenate = glue arrays end to end along an EXISTING axis (sizes may differ).")
p = np.array([2, 0, 2])
r = np.array([1, 0, 2])
show("p", p)
show("r", r)
show("np.stack([p, r])           as rows    (PQ train: pile of catalogues)", np.stack([p, r]))
show("np.stack([p, r], axis=1)   as columns (PQ encode: one row per vector)", np.stack([p, r], axis=1))
show("np.concatenate([p, r])     one long list (IVF search: gather candidates)", np.concatenate([p, r]))
left = np.array([[30, 30], [10, 10]])
right = np.array([[2, 2], [1, 1]])
show("np.concatenate([left, right], axis=1)   side by side (PQ decode)", np.concatenate([left, right], axis=1))
pause()

# ---------------------------------------------------------------------------
section(10, "Top-k: the k smallest, sorted",
         "argpartition = WHICH positions hold the k smallest (unsorted, fast).\n"
         " argsort      = how to SORT a few numbers.\n"
         " 'arg' always means: give me positions, not values.")
d = np.array([50, 10, 70, 30, 20, 90])
k = 3
show("d   (distances to vectors 0..5)", d)
part = np.argpartition(d, k - 1)[:k]
show("part = np.argpartition(d, k-1)[:k]   positions of the 3 smallest, any order", part)
show("d[part]                              their distances", d[part])
order = part[np.argsort(d[part])]
show("order = part[np.argsort(d[part])]    the same positions, sorted", order)
show("d[order]                             distances, nearest first", d[order])
print("\n  Answer: vectors", order, "with distances", d[order])
print("\n  flat.py and PQ do the same for many queries at once with take_along_axis;")
print("  IVF does it per query with plain indexing like here.")

print("\n" + "=" * 70)
print(" Done. Every index in VectorForge is built from these 10 moves.")
print("=" * 70)
