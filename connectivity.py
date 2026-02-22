import numpy as np


IDUM = (
  1, 6, 2, 32, 36, 37, 1, 2, 3, 33, 32, 38, 1, 3, 4, 34,
  33, 39, 1, 4, 5, 35, 34, 40, 1, 5, 6, 36, 35, 41, 7, 2, 6, 51,
  42, 37, 8, 3, 2, 47, 43, 38, 9, 4, 3, 48, 44, 39, 10, 5, 4,
  49, 45, 40, 11, 6, 5, 50, 46, 41, 8, 2, 12, 62, 47, 52, 9,
  3, 13, 63, 48, 53, 10, 4, 14, 64, 49, 54, 11, 5, 15, 65, 50,
  55, 7, 6, 16, 66, 51, 56, 7, 12, 2, 42, 57, 52, 8, 13, 3,
  43, 58, 53, 9, 14, 4, 44, 59, 54, 10, 15, 5, 45, 60, 55, 11,
  16, 6, 46, 61, 56, 8, 12, 18, 68, 62, 77, 9, 13, 19, 69, 63,
  78, 10, 14, 20, 70, 64, 79, 11, 15, 21, 71, 65, 80, 7, 16,
  17, 67, 66, 81, 7, 17, 12, 57, 67, 72, 8, 18, 13, 58, 68, 73,
  9, 19, 14, 59, 69, 74, 10, 20, 15, 60, 70, 75, 11, 21, 16,
  61, 71, 76, 22, 12, 17, 87, 82, 72, 23, 13, 18, 88, 83, 73,
  24, 14, 19, 89, 84, 74, 25, 15, 20, 90, 85, 75, 26, 16, 21,
  91, 86, 76, 22, 18, 12, 82, 92, 77, 23, 19, 13, 83, 93, 78,
  24, 20, 14, 84, 94, 79, 25, 21, 15, 85, 95, 80, 26, 17, 16,
  86, 96, 81, 22, 17, 27, 102, 87, 97, 23, 18, 28, 103, 88, 98,
  24, 19, 29, 104, 89, 99, 25, 20, 30, 105, 90, 100, 26, 21,
  31, 106, 91, 101, 22, 28, 18, 92, 107, 98, 23, 29, 19, 93,
  108, 99, 24, 30, 20, 94, 109, 100, 25, 31, 21, 95, 110, 101,
  26, 27, 17, 96, 111, 97, 22, 27, 28, 107, 102, 112, 23, 28,
  29, 108, 103, 113, 24, 29, 30, 109, 104, 114, 25, 30, 31,
  110, 105, 115, 26, 31, 27, 111, 106, 116, 122, 28, 27, 117,
  118, 112, 122, 29, 28, 118, 119, 113, 122, 30, 29, 119, 120,
  114, 122, 31, 30, 120, 121, 115, 122, 27, 31, 121, 117, 116
)
JVT1 = np.array(IDUM, dtype=int).reshape((6, 60), order="F") - 1


class Connectivity:
    def __init__(self, subtesserae):
        self._subtesserae = subtesserae

    def n0(self, itess, isubtess):
        if self._subtesserae == 1:
            return JVT1[0, itess]
        else:
            if isubtess == 0:
                return JVT1[0, itess]
            elif isubtess == 1:
                return JVT1[3, itess]
            elif isubtess == 2:
                return JVT1[3, itess]
            else:  # isubtess == 3
                return JVT1[1, itess]

    def n1(self, itess, isubtess):
        if self._subtesserae == 1:
            return JVT1[1, itess]
        else:
            if isubtess == 0:
                return JVT1[4, itess]
            elif isubtess == 1:
                return JVT1[5, itess]
            elif isubtess == 2:
                return JVT1[4, itess]
            else:  # isubtess == 3
                return JVT1[5, itess]

    def n2(self, itess, isubtess):
        if self._subtesserae == 1:
            return JVT1[2, itess]
        else:
            if isubtess == 0:
                return JVT1[3, itess]
            elif isubtess == 1:
                return JVT1[2, itess]
            elif isubtess == 2:
                return JVT1[5, itess]
            else:  # isubtess == 3
                return JVT1[4, itess]

