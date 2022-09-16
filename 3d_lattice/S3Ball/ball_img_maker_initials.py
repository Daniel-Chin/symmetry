COLOR3F = 'color3f'


COLOR3F_WHITE = (0.9, 0.9, 0.9)
COLOR3F_RED = (0.9, 0.05, 0.05)
COLOR3F_GREEN = (0.05, 0.9, 0.05)
COLOR3F_BLUE = (0.05, 0.05, 0.9)
COLOR3F_PINK = (0.9, 0.05, 0.9)
COLOR3F_YELLOW = (0.9, 0.9, 0.05)
COLOR3F_CYAN = (0.05, 0.9, 0.9)


S0 = 's0'
V0 = 'v0'
SAMPLE_WEIGHT = 'sample_weight'
"""
 7 | 8 | 9
——— ——— ———
 4 | 5 | 6
——— ——— ———
 1 | 2 | 3
"""
INIT_IDX = 'init_idx'
ENABLE = 'enable'


"""
 7 | 8 | 9
——— ——— ———
 4 | 5 | 6
——— ——— ———
 1 | 2 | 3
"""
NORMAL_DATA = [
    {
        INIT_IDX: 1,
        S0: [(-3, -1.5), (1, 1), (0, 3)],
        V0: [(0.2, 2), (-8, 8), (0.2, 3)],
        SAMPLE_WEIGHT: 1,
        ENABLE: True
    }, {
        INIT_IDX: 2,
        S0: [(-1.5, 1.5), (1, 1), (0, 2)],
        V0: [(-1, 1), (-8, 8), (0, 3)],
        SAMPLE_WEIGHT: 1,
        ENABLE: True
    }, {
        INIT_IDX: 3,
        S0: [(1.5, 3), (1, 1), (0, 3)],
        V0: [(-2, -0.2), (-8, 8), (0.2, 3)],
        SAMPLE_WEIGHT: 1,
        ENABLE: True
    }, {
        INIT_IDX: 4,
        S0: [(-5, -4), (1, 1), (4, 4.5)],
        V0: [(0.2, 2.5), (-8, 8), (-1, 1.)],
        SAMPLE_WEIGHT: 1,
        ENABLE: False
    }, {
        INIT_IDX: 5,
        S0: [(-0.5, 0.5), (1, 1), (4, 4.5)],
        V0: [(-1, 1), (-8, 8), (-1, 1)],
        SAMPLE_WEIGHT: 1,
        ENABLE: False
    }, {
        INIT_IDX: 6,
        S0: [(4, 5), (1, 1), (4, 4.5)],
        V0: [(-2.5, -0.2), (-8, 8), (-1, 1)],
        SAMPLE_WEIGHT: 1,
        ENABLE: False
    }, {
        INIT_IDX: 7,
        S0: [(-10, -8), (1, 1), (10, 12)],
        V0: [(0.5, 4), (-8, 8), (-3, -0.2)],
        SAMPLE_WEIGHT: 1,
        ENABLE: False
    }, {
        INIT_IDX: 8,
        S0: [(-1, 1), (1, 1), (10, 12)],
        V0: [(-1, 1), (-8, 8), (-3, 0)],
        SAMPLE_WEIGHT: 1,
        ENABLE: False
    }, {
        INIT_IDX: 9,
        S0: [(8, 12), (1, 1), (10, 12)],
        V0: [(-4, -0.5), (-8, 8), (-3, -0.2)],
        SAMPLE_WEIGHT: 1,
        ENABLE: False
    },
]


"""
 7 | 8 | 9
——— ——— ———
 4 | 5 | 6
——— ——— ———
 1 | 2 | 3
"""
HENTAI_EVAL_DATA = [
    {
        INIT_IDX: 1,
        S0: [(-3, -1.5), (1, 1), (0, 3)],
        V0: [(0.2, 2), (-8, 8), (0.2, 3)],
        SAMPLE_WEIGHT: 1,
        ENABLE: False
    }, {
        INIT_IDX: 2,
        S0: [(-1.5, 1.5), (1, 1), (0, 2)],
        V0: [(-1, 1), (-8, 8), (0, 3)],
        SAMPLE_WEIGHT: 1,
        ENABLE: False
    }, {
        INIT_IDX: 3,
        S0: [(1.5, 3), (1, 1), (0, 3)],
        V0: [(-2, -0.2), (-8, 8), (0.2, 3)],
        SAMPLE_WEIGHT: 1,
        ENABLE: False
    }, {
        INIT_IDX: 4,
        S0: [(-5, -4), (1, 1), (4, 4.5)],
        V0: [(0.2, 3), (-8, 8), (-1., 1.)],
        SAMPLE_WEIGHT: 1,
        ENABLE: True
    }, {
        INIT_IDX: 5,
        S0: [(-0.5, 0.5), (1, 1), (4, 4.5)],
        V0: [(-1, 1), (-8, 8), (-1., 1.)],
        SAMPLE_WEIGHT: 1,
        ENABLE: False
    }, {
        INIT_IDX: 6,
        S0: [(4, 5), (1, 1), (4, 4.5)],
        V0: [(-3, -0.2), (-8, 8), (-1., 1.)],
        SAMPLE_WEIGHT: 1,
        ENABLE: True
    }, {
        INIT_IDX: 7,
        S0: [(-8, -4), (1, 1), (8, 10)],
        V0: [(0.5, 3), (-8, 8), (-3, -0.2)],
        SAMPLE_WEIGHT: 1,
        ENABLE: True
    }, {
        INIT_IDX: 8,
        S0: [(-2, 2), (1, 1), (8, 10)],
        V0: [(-1, 1), (-8, 8), (-3, -0.2)],
        SAMPLE_WEIGHT: 1,
        ENABLE: True
    }, {
        INIT_IDX: 9,
        S0: [(4, 8), (1, 1), (8, 10)],
        V0: [(-3, -0.5), (-8, 8), (-3, -0.2)],
        SAMPLE_WEIGHT: 1,
        ENABLE: True
    },
]


COLOR_INIT_ALWAYS_GREEN = [
    {
        INIT_IDX: 1,
        COLOR3F: [(c, c) for c in COLOR3F_GREEN],
        SAMPLE_WEIGHT: 1,
        ENABLE: True
    }
]

COLOR_INIT_ALWAYS_WHITE = [
    {
        INIT_IDX: 1,
        COLOR3F: [(c, c) for c in COLOR3F_WHITE],
        SAMPLE_WEIGHT: 1,
        ENABLE: True
    }
]

COLOR_INIT_ALWAYS_RED = [
    {
        INIT_IDX: 1,
        COLOR3F: [(c, c) for c in COLOR3F_RED],
        SAMPLE_WEIGHT: 1,
        ENABLE: True
    }
]

COLOR_INIT_ALWAYS_BLUE = [
    {
        INIT_IDX: 1,
        COLOR3F: [(c, c) for c in COLOR3F_BLUE],
        SAMPLE_WEIGHT: 1,
        ENABLE: True
    }
]

COLOR_INIT_ALWAYS_PINK = [
    {
        INIT_IDX: 1,
        COLOR3F: [(c, c) for c in COLOR3F_PINK],
        SAMPLE_WEIGHT: 1,
        ENABLE: True
    }
]

COLOR_INIT_ALWAYS_YELLOW = [
    {
        INIT_IDX: 1,
        COLOR3F: [(c, c) for c in COLOR3F_YELLOW],
        SAMPLE_WEIGHT: 1,
        ENABLE: True
    }
]

COLOR_INIT_ALWAYS_CYAN = [
    {
        INIT_IDX: 1,
        COLOR3F: [(c, c) for c in COLOR3F_CYAN],
        SAMPLE_WEIGHT: 1,
        ENABLE: True
    }
]

COLOR_INIT_COLORFUL_6 = [
    {
        INIT_IDX: 1,
        COLOR3F: [(c, c) for c in COLOR3F_WHITE],
        SAMPLE_WEIGHT: 1,
        ENABLE: True
    }, {
        INIT_IDX: 2,
        COLOR3F: [(c, c) for c in COLOR3F_RED],
        SAMPLE_WEIGHT: 1,
        ENABLE: True
    }, {
        INIT_IDX: 3,
        COLOR3F: [(c, c) for c in COLOR3F_GREEN],
        SAMPLE_WEIGHT: 1,
        ENABLE: True
    }, {
        INIT_IDX: 4,
        COLOR3F: [(c, c) for c in COLOR3F_BLUE],
        SAMPLE_WEIGHT: 1,
        ENABLE: True
    }, {
        INIT_IDX: 5,
        COLOR3F: [(c, c) for c in COLOR3F_PINK],
        SAMPLE_WEIGHT: 1,
        ENABLE: True
    }, {
        INIT_IDX: 6,
        COLOR3F: [(c, c) for c in COLOR3F_YELLOW],
        SAMPLE_WEIGHT: 1,
        ENABLE: True
    }, {
        INIT_IDX: 7,
        COLOR3F: [(c, c) for c in COLOR3F_CYAN],
        SAMPLE_WEIGHT: 1,
        ENABLE: True
    }
]


COLOR_INIT_COLORFUL_CONTINUE = [
    {
        INIT_IDX: 1,
        COLOR3F: [(0, 1), (0, 1), (0, 1)],
        SAMPLE_WEIGHT: 1,
        ENABLE: True
    }
]
