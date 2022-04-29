'''
1->2, 1->3, 1->4, 1->5, 1->6,
2->3, 2->4, 2->5, 2->6,
3->4, 3->5, 3->6,
4->5, 4->6,
5->6
'''

import os
import numpy as np
from typing import Tuple, List

def get_index(i: int, j: int) -> int:
    if not i in range(1, 6) or not j in range(i + 1, 7) or j <= i:
        raise ValueError(f'The pair {(i, j)} is invalid for matching.')
    pairs = [(a, b) for a in range(1, 6) for b in range(a + 1, 7)]
    return pairs.index((i, j))

def get_pair(idx: int) -> Tuple[int, int]:
    if idx < 0 or idx >= 15:
        raise ValueError(f'The index {idx} is out of range for get_pair!')
    pairs = [(a, b) for a in range(1, 6) for b in range(a + 1, 7)]
    return pairs[idx]

def get_all_matches(path: str) -> List[np.ndarray]:
    all_matches = [[] for _ in range(15)]
    for file_name in sorted(os.listdir(path)):
        if not file_name.startswith('matching'):
            continue
        file_path = os.path.join(path, file_name)
        with open(file_path) as f:
            # get i from matching_i.txt
            i = int(file_name[-5])
            lines = f.readlines()
            assert len(lines) > 0
            num_features = int(lines[0].split(' ')[1])
            assert len(lines) == num_features + 1
            for line_idx in range(1, num_features + 1):
                line = lines[line_idx].strip()
                line = line.split(' ')
                num_matches = int(line[0])
                x1, y1 = float(line[4]), float(line[5])
                matches_in_line = [line[start:start+3] for start in range(6, len(line), 3)]
                assert len(matches_in_line) == num_matches - 1
                for match_in_line in matches_in_line:
                    assert len(match_in_line) == 3
                    j = int(match_in_line[0])
                    x2, y2 = float(match_in_line[1]), float(match_in_line[2])
                    all_matches[get_index(i, j)].append([[x1, y1], [x2, y2]])
    all_matches_final = []
    for matches in all_matches:
        matches_np = np.array(matches)
        all_matches_final.append(matches_np)
    return all_matches_final
