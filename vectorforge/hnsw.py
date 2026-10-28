"""HNSW: Hierarchical Navigable Small World graph.  <-- YOU write 4 methods, in this order:
    1. _random_level   2. _search_layer   3. _insert   4. search
(tests/test_hnsw.py has a -k filter for each step.)


To make the walk short, the graph has LAYERS, like a map with zoom levels:
  - layer 0 holds EVERY vector, each linked to its close neighbours (street level)
  - layer 1 holds about 1 in M vectors (city level), layer 2 about 1 in M^2 (country level), ...
  - search starts at the top (few nodes, long jumps), walks greedily to the node
    closest to the query, drops one layer, continues from there, ... and finishes
    with a careful search on layer 0.

What the index stores (full vectors, like Flat and IVF: distances are exact):
  self.xb            (n, d)  all vectors
  self.levels        list    levels[i] = the top layer of vector i
  self.graph         list of dicts, one per layer:
                     self.graph[l][i] = list of the ids linked to i on layer l
                     (a vector on layer l is in self.graph[0], ..., self.graph[l])
  self.entry_point   int     where every search starts: a vector on the top layer
  self.max_level     int     the top layer number (= len(self.graph) - 1)

Settings:
  M                links per node (M0 = 2 * M on layer 0)
  ef_construction  how many candidates to keep while inserting (bigger = better graph, slower build)
  ef_search        how many candidates to keep while searching (bigger = higher recall, slower search)

Goal: make tests/test_hnsw.py pass. Squared L2 only.
"""

import heapq
import math

import numpy as np


class HNSWIndex:
    def __init__(self, M: int = 16, ef_construction: int = 100, ef_search: int = 50, seed: int = 0):
        self.M = M                       # max links per node on layers 1, 2, ...
        self.M0 = 2 * M                  # max links per node on layer 0
        self.mL = 1 / math.log(M)        # level multiplier: about 1 in M nodes go up a layer
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.rng = np.random.default_rng(seed)
        self.xb = None                   # (n, d) float32
        self.levels = []                 # levels[i] = top layer of vector i
        self.graph = []                  # graph[l][i] = neighbour ids of i on layer l
        self.entry_point = None          # id of a vector on the top layer
        self.max_level = -1              # top layer number

    # ------------------------------------------------------------- given helpers
    def _distances(self, q: np.ndarray, ids) -> np.ndarray:
        """Squared distances from one vector q (d,) to the vectors with these ids. (Given.)"""
        return ((self.xb[ids] - q) ** 2).sum(axis=1)

    def _shrink(self, node: int, level: int) -> None:
        """If node has too many links on this layer, keep only its closest ones. (Given.)"""
        limit = self.M0 if level == 0 else self.M
        nbrs = self.graph[level][node]
        if len(nbrs) > limit:
            d = self._distances(self.xb[node], nbrs)
            keep = np.argsort(d)[:limit]
            self.graph[level][node] = [nbrs[j] for j in keep]

    def add(self, xb: np.ndarray) -> None:
        """Store the vectors and insert them into the graph one by one. (Given.)"""
        xb = np.asarray(xb, dtype=np.float32)
        start = 0 if self.xb is None else len(self.xb)
        self.xb = xb if self.xb is None else np.vstack([self.xb, xb])
        for i in range(start, len(self.xb)):
            self._insert(i)

    # ------------------------------------------------------------- step 1
    def _random_level(self) -> int:
        """Draw the top layer for a new vector.

        level = floor( -ln(U) * mL ),  U uniform in (0, 1]
        Most draws give 0; about 1 in M gives 1 or more; about 1 in M^2 gives 2 or more.

        Hint: self.rng.random() is uniform in [0, 1), so 1 - self.rng.random() is in (0, 1]
        (never 0, so the log is safe). math.log is ln. int(...) rounds down for positives.
        """
        u = 1.0-self.rng.random()
        mL = 1 / np.log(self.M)
        return int(-(np.log(u))*mL)

    # ------------------------------------------------------------- step 2
    def _search_layer(self, q: np.ndarray, entry_points: list, ef: int, level: int) -> list:
        """Best-first search on ONE layer. Returns up to ef (distance, id) pairs, closest first.

        Keep three things:
          visited     a set of ids already measured (never measure a node twice)
          candidates  a MIN-heap of (distance, id): nodes still to explore, closest on top
          results     a MAX-heap of (-distance, id): the ef best found so far, WORST on top
                      (heapq only does min-heaps; storing -distance makes the worst come out first)

        Start: measure the entry points, put them in visited, candidates and results.
        Loop while candidates is not empty:
          1. pop the closest candidate c
          2. if c is farther than the worst result, stop: nothing closer is left to find
          3. look at c's neighbours on this layer that are not visited yet;
             mark them visited and measure them (self._distances)
          4. each one that is closer than the worst result (or while results has fewer than ef):
             push it into candidates AND results; if results now has more than ef, pop the worst
        Return the results as (distance, id) pairs sorted closest first.

        heapq in one line each:
          heapq.heappush(h, (d, i))   add      heapq.heappop(h)   remove + return the smallest
          h[0]                        the smallest, without removing it
        """
        visited  = set(entry_points)
        d= self._distances(q,entry_points)
        candidates = [(float(di),i) for di, i in zip(d,entry_points)]
        heapq.heapify(candidates)
        results = [(-float(di),i) for di, i  in zip(d,entry_points)]
        heapq.heapify(results)
        
        while len(results)>ef:
            heapq.headppop(results)
            
        while candidates:
            dist_c, c = heapq.heappop(candidates)
            worst = -results[0][0]
            
            if dist_c > worst:
                break
            new = [n for n in self.graph[level][c] if n not in visited]
            visited.update(new)
            
            if not new:
                continue
            
            dists = self._distances(q,new)
            for dist, n in zip(dists,new):
                dist = float(dist)
                if len(results) < ef or dist < -results[0][0]:
                    heapq.heappush(candidates ,(dist,n))
                    heapq.heappush(results,(-dist,n))
                    if len(results)>ef:
                        heapq.heappop(results)
                        
        return sorted((-neg_d,i) for neg_d,i in results) 
            
            
            

    # ------------------------------------------------------------- step 3
    def _insert(self, i: int) -> None:
        """Insert vector i (already in self.xb) into the graph.

          1. level = a random level; append it to self.levels
          2. make sure self.graph has a dict for every layer 0..level,
             and give i an empty neighbour list on each of those layers
          3. if the graph was empty (no entry point yet): i becomes the entry point,
             max_level = level, done
          4. zoom in: from the top layer down to level + 1, walk greedily
             (_search_layer with ef = 1) and keep the closest node as the next entry
          5. for each layer from min(level, max_level) down to 0:
               found = _search_layer(..., self.ef_construction, layer)
               link i to the closest M of them (M0 on layer 0): both directions,
               and call self._shrink on each neighbour (it may now have too many links)
               use all of found as the entry points for the next layer down
          6. if level > max_level: i becomes the new entry point and max_level = level
        """
        q = self.xb[i]
        
        level = self._random_level()
        self.levels.append(level)
        
        while len(self.graph) <= level:
            self.graph.append({})
        for l in range(level+1):
            self.graph[l][i] = []
            
        if self.entry_point is None:
            self.entry_point, self.max_level = i,level
            return
        
        ep = [self.entry_point]
        for l in range(self.max_level, level, -1):
            ep = [self._search_layer(q, ep, 1,l)[0][1]]
            
        for l in range(min(level,self.max_level), -1, -1):
            found = self._search_layer(q,ep,self.ef_construction,l)
            limit = self.M0 if l == 0 else self.M
            neighbours = [n for _, n  in found[:limit]]
            self.graph[l][i] = neighbours
            for n in neighbours:
                self.graph[l][n].append(i)
                self._shrink(n,l)
            ep = [n for _,n in found]
            
        if level > self.max_level:
            self.entry_point, self.max_level = i,level

    # ------------------------------------------------------------- step 4
    def search(self, xq: np.ndarray, k: int):
        """Return (distances, ids), both (nq, k), exact squared L2, nearest first.

        For each query: start at the entry point, zoom in greedily from the top layer
        down to layer 1 (ef = 1, like step 4 of _insert), then _search_layer on layer 0
        with ef = max(self.ef_search, k) and keep the first k.
        Pad with -1 / inf if fewer than k were found (e.g. fewer than k vectors stored).
        """
        xq = np.asarray(xq, dtype=np.float32)
        nq = len(xq)

        # empty answer sheets: -1 / inf mean "nothing found" (same as IVF)
        D = np.full((nq, k), np.inf, dtype=np.float32)
        I = np.full((nq, k), -1, dtype=np.int64)
        if self.entry_point is None:                 # empty index: nothing to find
            return D, I

        for qi in range(nq):
            q = xq[qi]

            # 1. start at the entry point
            ep = [self.entry_point]

            # 2. zoom in: from the top layer down to layer 1, greedily (ef = 1)
            for l in range(self.max_level, 0, -1):
                ep = [self._search_layer(q, ep, 1,l)[0][1]]

            # 3. careful search on layer 0, keep the k closest
            found = self._search_layer(q, ep, max(self.ef_search, k), 0)[:k]

            # 4. write them into row qi of the answer sheets
            kk = len(found)
            D[qi, :kk] = [d for d, _ in found]
            I[qi, :kk] = [n for _, n in found]

        return D, I
