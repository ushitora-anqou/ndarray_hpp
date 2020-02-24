#include "ndarray.hpp"

void test_ndarray()
{
    using namespace ndarray_hpp;

    static const auto iota234f = make_ndarray<float, 2, 3, 4>(
        {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
         13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
#define FOREACH234                  \
    for (int i = 0; i < 2; i++)     \
        for (int j = 0; j < 3; j++) \
            for (int k = 0; k < 4; k++)

    {
        auto ndary = iota234f;
        {
            auto shape_ans = std::array<size_t, 3>{2, 3, 4};
            assert(ndary.shape() == shape_ans);
        }
        assert(ndary.dim() == 3);
        assert(ndary.size() == 24);
        assert(ndary(1, 2, 3) == 24);
        assert(ndary.reshape<24>()(22) == 23);
        int cnt = 0;
        ndary.each([&](auto i, auto j, auto k) {
            cnt++;
            assert(ndary(i, j, k) ==
                   static_cast<float>(i * 12 + j * 4 + k + 1));
        });
        assert(cnt == 2 * 3 * 4);
    }

    {
        auto ndary = ndarray<int, 2, 3, 4>::ones();
        FOREACH234
        {
            assert(ndary(i, j, k) == 1);
        }
    }

    {
        auto ndary = ndarray<int, 2, 3, 4>::zeros();
        FOREACH234
        {
            assert(ndary(i, j, k) == 0);
        }
    }

    {
        auto ndary = ndarray<float, 2, 3, 4>::randn(std::mt19937{0}, 0, 1);
        std::mt19937 rand{0};
        std::normal_distribution<> dist{0, 1};
        FOREACH234
        {
            assert(ndary(i, j, k) == static_cast<float>(dist(rand)));
        }
    }

    {
        auto ndary = iota234f;
        ndary.mask_i(
            [&](auto i, auto j, auto k) { return (i + j + k) % 2 == 0; });
        FOREACH234
        {
            assert(ndary(i, j, k) ==
                   ((i + j + k) % 2 == 0 ? (i * 12 + j * 4 + k + 1) : 0));
        }
    }

    {
        auto ndary = iota234f;
        ndary.mask_v(
            [&](auto val) { return (static_cast<int>(val) % 2) == 0; });
        FOREACH234
        {
            assert(ndary(i, j, k) == ((i * 12 + j * 4 + k + 1) % 2 == 0
                                          ? (i * 12 + j * 4 + k + 1)
                                          : 0));
        }
    }

    {
        auto ndary = iota234f;
        ndary.map_i([&](auto i, auto j, auto k) {
            return (i + j + k) % 2 == 0 ? ndary(i, j, k) : 0;
        });
        FOREACH234
        {
            assert(ndary(i, j, k) ==
                   ((i + j + k) % 2 == 0 ? (i * 12 + j * 4 + k + 1) : 0));
        }
    }

    {
        auto ndary = iota234f;
        ndary.map_v([&](auto val) {
            return (static_cast<int>(val) % 2) == 0 ? val : 0;
        });
        FOREACH234
        {
            assert(ndary(i, j, k) == ((i * 12 + j * 4 + k + 1) % 2 == 0
                                          ? (i * 12 + j * 4 + k + 1)
                                          : 0));
        }
    }

    {
        auto ndary0 = iota234f;
        auto ndary1 = iota234f;
        ndary0 += ndary0;
        FOREACH234
        {
            assert(ndary0(i, j, k) == ndary1(i, j, k) * 2);
        }
    }

    {
        auto ndary0 = iota234f;
        auto ndary1 = iota234f;
        ndary0 += ndary1;
        FOREACH234
        {
            assert(ndary0(i, j, k) == ndary1(i, j, k) * 2);
        }
    }

    {
        auto ndary0 = iota234f;
        auto ndary1 = iota234f;
        ndary0 += 1;
        FOREACH234
        {
            assert(ndary0(i, j, k) == ndary1(i, j, k) + 1);
        }
    }

    {
        auto ndary0 = iota234f;
        auto ndary1 = iota234f;
        ndary0 -= ndary1;
        FOREACH234
        {
            assert(ndary0(i, j, k) == 0);
        }
    }

    {
        auto ndary0 = iota234f;
        auto ndary1 = iota234f;
        ndary0 -= 1;
        FOREACH234
        {
            assert(ndary0(i, j, k) == ndary1(i, j, k) - 1);
        }
    }

    {
        auto ndary0 = iota234f;
        auto ndary1 = iota234f;
        ndary0 *= ndary1;
        FOREACH234
        {
            assert(ndary0(i, j, k) == ndary1(i, j, k) * ndary1(i, j, k));
        }
    }

    {
        auto ndary0 = iota234f;
        auto ndary1 = iota234f;
        ndary0 *= 2;
        FOREACH234
        {
            assert(ndary0(i, j, k) == ndary1(i, j, k) * 2);
        }
    }

    {
        auto ndary0 = iota234f;
        auto ndary1 = iota234f;
        ndary0 /= 2;
        FOREACH234
        {
            assert(ndary0(i, j, k) == ndary1(i, j, k) / 2);
        }
    }

    {
        auto ndary0 = iota234f;
        auto ndary1 = iota234f;
        ndary0 /= ndary1;
        FOREACH234
        {
            assert(ndary0(i, j, k) == 1);
        }
    }

    {
        auto ndary0 = iota234f;
        auto ndary1 = iota234f;
        auto ndary2 = ndary0 + ndary1;
        FOREACH234
        {
            assert(ndary2(i, j, k) == ndary0(i, j, k) + ndary1(i, j, k));
        }
    }

    {
        auto ndary0 = make_ndarray<float, 2, 3, 4>(
            {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
             13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
        auto ndary1 = make_ndarray<float, 2, 1, 4>({1, 2, 3, 4, 5, 6, 7, 8});
        auto ndary2 = ndary0 + ndary1;
        FOREACH234
        {
            assert(ndary2(i, j, k) == ndary0(i, j, k) + ndary1(i, 0, k));
        }
    }

    {
        auto ndary0 = make_ndarray<float, 2, 3, 1>({1, 2, 3, 4, 5, 6});
        auto ndary1 = make_ndarray<float, 1, 4>({1, 2, 3, 4});

        {
            auto ndary = ndary0 + ndary1;
            FOREACH234
            {
                assert(ndary(i, j, k) == ndary0(i, j, 0) + ndary1(0, k));
            }
        }

        {
            auto ndary = ndary0 - ndary1;
            FOREACH234
            {
                assert(ndary(i, j, k) == ndary0(i, j, 0) - ndary1(0, k));
            }
        }

        {
            auto ndary = ndary0 * ndary1;
            FOREACH234
            {
                assert(ndary(i, j, k) == ndary0(i, j, 0) * ndary1(0, k));
            }
        }

        {
            auto ndary = ndary0 / ndary1;
            FOREACH234
            {
                assert(ndary(i, j, k) == ndary0(i, j, 0) / ndary1(0, k));
            }
        }
    }

    {
        auto ndary0 = iota234f;

        {
            auto ndary = ndary0 + 3;
            FOREACH234
            {
                assert(ndary(i, j, k) == ndary0(i, j, k) + 3);
            }
        }

        {
            auto ndary = ndary0 - 3;
            FOREACH234
            {
                assert(ndary(i, j, k) == ndary0(i, j, k) - 3);
            }
        }

        {
            auto ndary = ndary0 * 3;
            FOREACH234
            {
                assert(ndary(i, j, k) == ndary0(i, j, k) * 3);
            }
        }

        {
            auto ndary = ndary0 / 3;
            FOREACH234
            {
                assert(ndary(i, j, k) == ndary0(i, j, k) / 3);
            }
        }
    }

    {
        auto lhs = iota234f;
        auto rhs = iota234f;
        assert(lhs == rhs);
        lhs(1, 0, 3) += 2;
        assert(lhs != rhs);
    }

    {
        auto ndary = iota234f;
        auto ndary0 = make_ndarray<float, 3, 4>(
            {14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36});
        auto ndary1 =
            make_ndarray<float, 2, 4>({15, 18, 21, 24, 51, 54, 57, 60});
        auto ndary2 = make_ndarray<float, 2, 3>({10, 26, 42, 58, 74, 90});
        assert(sum<0>(ndary) == ndary0);
        assert(sum<1>(ndary) == ndary1);
        assert(sum<2>(ndary) == ndary2);
    }

    {
        auto ndary = make_ndarray<int, 2, 3, 4>(
            {13, 15, 21, 6, 17, 18, 12, 16, 14, 10, 4, 0,
             1,  7,  11, 3, 9,  2,  8,  19, 20, 23, 5, 22});
        auto ndary0 =
            make_ndarray<size_t, 3, 4>({0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1});
        auto ndary1 = make_ndarray<size_t, 2, 4>({1, 1, 0, 1, 2, 2, 0, 2});
        auto ndary2 = make_ndarray<size_t, 2, 3>({2, 1, 0, 2, 3, 1});
        assert(argmax(ndary) == 21);
        assert(argmax<0>(ndary) == ndary0);
        assert(argmax<1>(ndary) == ndary1);
        assert(argmax<2>(ndary) == ndary2);
    }

    {
        auto ndary = make_ndarray<int, 2, 3, 4>(
            {13, 15, 21, 6, 17, 18, 12, 16, 14, 10, 4, 0,
             1,  7,  11, 3, 9,  2,  8,  19, 20, 23, 5, 22});
        auto ndary0 = make_ndarray<int, 3, 4>(
            {13, 15, 21, 6, 17, 18, 12, 19, 20, 23, 5, 22});
        auto ndary1 = make_ndarray<int, 2, 4>({17, 18, 21, 16, 20, 23, 11, 22});
        auto ndary2 = make_ndarray<int, 2, 3>({21, 18, 14, 11, 19, 23});
        assert(max(ndary) == 23);
        assert(max<0>(ndary) == ndary0);
        assert(max<1>(ndary) == ndary1);
        assert(max<2>(ndary) == ndary2);
    }

    {
        auto ndary0 = make_ndarray<int, 2, 3>({1, 4, 9, 16, 25, 36});
        auto ndary1 = make_ndarray<int, 2, 3>({1, 2, 3, 4, 5, 6});
        assert(sqrt(ndary0) == ndary1);
        assert(pow(ndary1, 2) == ndary0);
    }

    {
        auto ndary0 = iota234f;
        auto ndary1 = make_ndarray<float, 4, 3, 2>(
            {1, 13, 5, 17, 9,  21, 2, 14, 6, 18, 10, 22,
             3, 15, 7, 19, 11, 23, 4, 16, 8, 20, 12, 24});
        assert(transpose(ndary0) == ndary1);
    }

#undef FOREACH234
}

int main()
{
    test_ndarray();
}
