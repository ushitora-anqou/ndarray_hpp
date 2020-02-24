#ifndef ANQOU_NDARRAY_HPP
#define ANQOU_NDARRAY_HPP

#include <array>
#include <cassert>
#include <random>
#include <tuple>
#include <vector>

//
#include <tao/seq/concatenate.hpp>
#include <tao/seq/make_integer_range.hpp>
#include <tao/seq/reverse.hpp>

namespace ndarray_hpp {
template <class Float = float, size_t... Shape>
struct ndarray {
    static constexpr auto dim()
    {
        return sizeof...(Shape);
    }
    static constexpr auto shape()
    {
        return std::array<size_t, sizeof...(Shape)>{Shape...};
    }
    static constexpr auto size()
    {
        return (Shape * ...);
    }

    static_assert(size() > 0);

    std::vector<Float> data;

    ndarray() : data((Shape * ...))
    {
    }

    ndarray(std::vector<Float> data) : data(std::move(data))
    {
        assert(size() == this->data.size());
    }

    static constexpr auto full(Float val)
    {
        return ndarray<Float, Shape...>(std::vector<Float>(size(), val));
    }

    static constexpr auto zeros()
    {
        return full(0);
    }

    static constexpr auto ones()
    {
        return full(1);
    }

    template <class RandEngine>
    static auto randn(RandEngine&& engine, Float mean, Float std)
    {
        std::normal_distribution<> dist{mean, std};
        ndarray<Float, Shape...> ret;
        ret.each([&](auto&&... args) { ret(args...) = dist(engine); });
        return ret;
    }

    template <class... Indices>
    const Float& operator()(Indices&&... indices) const
    {
        return data.at(
            index(shape(), std::make_tuple(std::forward<Indices>(indices)...)));
    }

    template <class... Indices>
    Float& operator()(Indices&&... indices)
    {
        return const_cast<Float&>(static_cast<const ndarray&>(*this)(
            std::forward<Indices>(indices)...));
    }

    template <size_t... NewShape>
    auto reshape() const
    {
        static_assert(size() == ndarray<Float, NewShape...>::size());
        return ndarray<Float, NewShape...>{data};
    }

    template <size_t N = 0, class Func>
    void each(Func func) const
    {
        if constexpr (N < dim()) {
            for (size_t i = 0; i < shape().at(N); i++)
                each<N + 1>([i, func](auto&&... args) {
                    return func(i, std::forward<decltype(args)>(args)...);
                });
        }
        else {
            func();
        }
    }

    template <class Pred>
    void mask_i(Pred pred)
    {
        each([&](auto&&... indices) {
            if (!pred(std::forward<decltype(indices)>(indices)...))
                (*this)(std::forward<decltype(indices)>(indices)...) = 0;
        });
    }

    template <class Pred>
    void mask_v(Pred pred)
    {
        each([&](auto&&... indices) {
            auto& val = (*this)(std::forward<decltype(indices)>(indices)...);
            if (!pred(val))
                val = 0;
        });
    }

    template <class Func>
    void map_i(Func func)
    {
        each([&](auto&&... indices) {
            auto& val = (*this)(std::forward<decltype(indices)>(indices)...);
            val = func(std::forward<decltype(indices)>(indices)...);
        });
    }

    template <class Func>
    void map_v(Func func)
    {
        each([&](auto&&... indices) {
            auto& val = (*this)(std::forward<decltype(indices)>(indices)...);
            val = func(val);
        });
    }

    ndarray<Float, Shape...>& operator+=(const ndarray<Float, Shape...>& rhs)
    {
        for (size_t i = 0; i < size(); i++)
            data[i] += rhs.data[i];
        return *this;
    }

    ndarray<Float, Shape...>& operator+=(Float rhs)
    {
        for (size_t i = 0; i < size(); i++)
            data[i] += rhs;
        return *this;
    }

    ndarray<Float, Shape...>& operator-=(const ndarray<Float, Shape...>& rhs)
    {
        for (size_t i = 0; i < size(); i++)
            data[i] -= rhs.data[i];
        return *this;
    }

    ndarray<Float, Shape...>& operator-=(Float rhs)
    {
        for (size_t i = 0; i < size(); i++)
            data[i] -= rhs;
        return *this;
    }

    ndarray<Float, Shape...>& operator*=(const ndarray<Float, Shape...>& rhs)
    {
        for (size_t i = 0; i < size(); i++)
            data[i] *= rhs.data[i];
        return *this;
    }

    ndarray<Float, Shape...>& operator*=(Float rhs)
    {
        for (size_t i = 0; i < size(); i++)
            data[i] *= rhs;
        return *this;
    }

    ndarray<Float, Shape...>& operator/=(const ndarray<Float, Shape...>& rhs)
    {
        for (size_t i = 0; i < size(); i++)
            data[i] /= rhs.data[i];
        return *this;
    }

    ndarray<Float, Shape...>& operator/=(Float rhs)
    {
        for (size_t i = 0; i < size(); i++)
            data[i] /= rhs;
        return *this;
    }

private:
    template <size_t N = 0, size_t Size, class... Types2>
    size_t index(std::array<size_t, Size> shape, std::tuple<Types2...> indices,
                 size_t sum = 0) const
    {
        static_assert(Size == sizeof...(Types2));
        if constexpr (N < Size) {
            return index<N + 1>(shape, indices,
                                sum * shape.at(N) + std::get<N>(indices));
        }
        else {
            return sum;
        }
    }
};

template <size_t LN = 0, class Float, class Func, class LhsPicker,
          class RhsPicker, size_t... LShape, size_t... RShape>
void broadcast(const ndarray<Float, LShape...>& lhs,
               const ndarray<Float, RShape...>& rhs, Func func,
               LhsPicker lhs_picker, RhsPicker rhs_picker)
{
    constexpr auto lhs_dim = ndarray<Float, LShape...>::dim();
    constexpr auto rhs_dim = ndarray<Float, RShape...>::dim();
    constexpr auto lhs_shape = ndarray<Float, LShape...>::shape();
    constexpr auto rhs_shape = ndarray<Float, RShape...>::shape();

    static_assert(lhs_dim >= rhs_dim);  // FIXME: relax this condition

    if constexpr (LN < lhs_dim) {
        constexpr auto RN = LN - lhs_dim + rhs_dim;
        if constexpr (RN >=
                      rhs_dim /* RN is unsigned, so this means RN < 0 */) {
            for (size_t i = 0; i < lhs.shape().at(LN); i++)
                broadcast<LN + 1>(
                    lhs, rhs,
                    [i, func](auto&&... args) {
                        return func(i, std::forward<decltype(args)>(args)...);
                    },
                    [i, lhs_picker](auto&&... args) {
                        return lhs_picker(
                            i, std::forward<decltype(args)>(args)...);
                    },
                    rhs_picker);
        }
        else if constexpr (lhs_shape.at(LN) == rhs_shape.at(RN)) {
            for (size_t i = 0; i < lhs.shape().at(LN); i++)
                broadcast<LN + 1>(
                    lhs, rhs,
                    [i, func](auto&&... args) {
                        return func(i, std::forward<decltype(args)>(args)...);
                    },
                    [i, lhs_picker](auto&&... args) {
                        return lhs_picker(
                            i, std::forward<decltype(args)>(args)...);
                    },
                    [i, rhs_picker](auto&&... args) {
                        return rhs_picker(
                            i, std::forward<decltype(args)>(args)...);
                    });
        }
        else if constexpr (rhs_shape.at(RN) == 1) {
            for (size_t i = 0; i < lhs.shape().at(LN); i++)
                broadcast<LN + 1>(
                    lhs, rhs,
                    [i, func](auto&&... args) {
                        return func(i, std::forward<decltype(args)>(args)...);
                    },
                    [i, lhs_picker](auto&&... args) {
                        return lhs_picker(
                            i, std::forward<decltype(args)>(args)...);
                    },
                    [rhs_picker](auto&&... args) {
                        return rhs_picker(
                            0, std::forward<decltype(args)>(args)...);
                    });
        }
        else if constexpr (lhs_shape.at(LN) == 1) {
            for (size_t i = 0; i < rhs.shape().at(RN); i++)
                broadcast<LN + 1>(
                    lhs, rhs,
                    [i, func](auto&&... args) {
                        return func(i, std::forward<decltype(args)>(args)...);
                    },
                    [lhs_picker](auto&&... args) {
                        return lhs_picker(
                            0, std::forward<decltype(args)>(args)...);
                    },
                    [i, rhs_picker](auto&&... args) {
                        return rhs_picker(
                            i, std::forward<decltype(args)>(args)...);
                    });
        }
    }
    else {
        func()(lhs_picker(), rhs_picker());
    }
}

template <size_t LN = 0, class Float, class Func, size_t... LShape,
          size_t... RShape>
void broadcast(const ndarray<Float, LShape...>& lhs,
               const ndarray<Float, RShape...>& rhs, Func func)
{
    broadcast(
        lhs, rhs,
        [func](auto&&... indices) {
            return [indices..., func](auto&& lhs, auto&& rhs) {
                func(lhs, rhs, indices...);
            };
        },
        [&lhs](auto&&... indices) {
            return lhs(std::forward<decltype(indices)>(indices)...);
        },
        [&rhs](auto&&... indices) {
            return rhs(std::forward<decltype(indices)>(indices)...);
        });
}

namespace detail {
template <size_t LN = 0, class Float, size_t... LShape, size_t... RShape>
auto broadcasted_t_impl(const ndarray<Float, LShape...>& lhs,
                        const ndarray<Float, RShape...>& rhs)
{
    constexpr auto lhs_dim = ndarray<Float, LShape...>::dim();
    constexpr auto rhs_dim = ndarray<Float, RShape...>::dim();
    constexpr auto lhs_shape = ndarray<Float, LShape...>::shape();
    constexpr auto rhs_shape = ndarray<Float, RShape...>::shape();

    static_assert(lhs_dim >= rhs_dim);  // FIXME: relax this condition

    if constexpr (LN < lhs_dim) {
        constexpr auto RN = LN - lhs_dim + rhs_dim;
        if constexpr (RN >=
                      rhs_dim /* RN is unsigned, so this means RN < 0 */) {
            return tao::seq::concatenate_t<
                tao::seq::integer_sequence<size_t, lhs_shape.at(LN)>,
                decltype(broadcasted_t_impl<LN + 1>(lhs, rhs))>();
        }
        else if constexpr (lhs_shape.at(LN) == rhs_shape.at(RN)) {
            return tao::seq::concatenate_t<
                tao::seq::integer_sequence<size_t, lhs_shape.at(LN)>,
                decltype(broadcasted_t_impl<LN + 1>(lhs, rhs))>();
        }
        else if constexpr (rhs_shape.at(RN) == 1) {
            return tao::seq::concatenate_t<
                tao::seq::integer_sequence<size_t, lhs_shape.at(LN)>,
                decltype(broadcasted_t_impl<LN + 1>(lhs, rhs))>();
        }
        else if constexpr (lhs_shape.at(LN) == 1) {
            return tao::seq::concatenate_t<
                tao::seq::integer_sequence<size_t, rhs_shape.at(RN)>,
                decltype(broadcasted_t_impl<LN + 1>(lhs, rhs))>();
        }
    }
    else {
        return tao::seq::make_integer_range<size_t, 0, 0>();
    }
}
template <class, class>
struct seq2ndarray {
};
template <class Float, size_t... Shape>
struct seq2ndarray<Float, std::integer_sequence<size_t, Shape...>> {
    using type = ndarray<Float, Shape...>;
};
}  // namespace detail

template <class Float, class LHS, class RHS>
using broadcasted_t =
    typename detail::seq2ndarray<Float, decltype(detail::broadcasted_t_impl(
                                            std::declval<LHS>(),
                                            std::declval<RHS>()))>::type;

template <class Float, size_t... LShape, size_t... RShape>
inline auto operator+(const ndarray<Float, LShape...>& lhs,
                      const ndarray<Float, RShape...>& rhs)
{
    broadcasted_t<Float, decltype(lhs), decltype(rhs)> res;
    broadcast(lhs, rhs, [&](auto&& lhs, auto&& rhs, auto&&... indices) {
        res(indices...) = lhs + rhs;
    });
    return res;
}

template <class Float, class RHS, size_t... LShape>
inline auto operator+(const ndarray<Float, LShape...>& lhs, RHS rhs)
{
    return lhs + ndarray<Float, 1>{std::vector<Float>{static_cast<Float>(rhs)}};
}

template <class Float, size_t... LShape, size_t... RShape>
inline auto operator-(const ndarray<Float, LShape...>& lhs,
                      const ndarray<Float, RShape...>& rhs)
{
    broadcasted_t<Float, decltype(lhs), decltype(rhs)> res;
    broadcast(lhs, rhs, [&](auto&& lhs, auto&& rhs, auto&&... indices) {
        res(indices...) = lhs - rhs;
    });
    return res;
}

template <class Float, class RHS, size_t... LShape>
inline auto operator-(const ndarray<Float, LShape...>& lhs, RHS rhs)
{
    return lhs - ndarray<Float, 1>{std::vector<Float>{static_cast<Float>(rhs)}};
}

template <class Float, size_t... LShape, size_t... RShape>
inline auto operator*(const ndarray<Float, LShape...>& lhs,
                      const ndarray<Float, RShape...>& rhs)
{
    broadcasted_t<Float, decltype(lhs), decltype(rhs)> res;
    broadcast(lhs, rhs, [&](auto&& lhs, auto&& rhs, auto&&... indices) {
        res(indices...) = lhs * rhs;
    });
    return res;
}

template <class Float, class RHS, size_t... LShape>
inline auto operator*(const ndarray<Float, LShape...>& lhs, RHS rhs)
{
    return lhs * ndarray<Float, 1>{std::vector<Float>{static_cast<Float>(rhs)}};
}

template <class Float, size_t... LShape, size_t... RShape>
inline auto operator/(const ndarray<Float, LShape...>& lhs,
                      const ndarray<Float, RShape...>& rhs)
{
    broadcasted_t<Float, decltype(lhs), decltype(rhs)> res;
    broadcast(lhs, rhs, [&](auto&& lhs, auto&& rhs, auto&&... indices) {
        res(indices...) = lhs / rhs;
    });
    return res;
}

template <class Float, class RHS, size_t... LShape>
inline auto operator/(const ndarray<Float, LShape...>& lhs, RHS rhs)
{
    return lhs / ndarray<Float, 1>{std::vector<Float>{static_cast<Float>(rhs)}};
}

template <class Float, size_t... Shape>
inline bool operator==(const ndarray<Float, Shape...>& lhs,
                       const ndarray<Float, Shape...>& rhs)
{
    for (size_t i = 0; i < lhs.size(); i++)
        if (lhs.data[i] != rhs.data[i])
            return false;
    return true;
}

template <class Float, size_t... Shape>
inline bool operator!=(const ndarray<Float, Shape...>& lhs,
                       const ndarray<Float, Shape...>& rhs)
{
    return !(lhs == rhs);
}

template <class Float, size_t... Shape>
inline ndarray<Float, Shape...> operator-(const ndarray<Float, Shape...>& src)
{
    ndarray<Float, Shape...> ret;
    ret.each([&](auto&&... indices) {
        ret(std::forward<decltype(indices)>(indices)...) =
            -src(std::forward<decltype(indices)>(indices)...);
    });
    return ret;
}

template <class Float, size_t... Shape>
inline ndarray<Float, Shape...> sqrt(ndarray<Float, Shape...> src)
{
    src.map_v([](auto&& val) {
        using std::sqrt;
        return sqrt(val);
    });
    return src;
}

template <class Float, size_t... Shape>
inline ndarray<Float, Shape...> log(ndarray<Float, Shape...> src)
{
    src.map_v([](auto&& val) {
        using std::log;
        return log(val);
    });
    return src;
}

template <class Float, size_t... Shape>
inline ndarray<Float, Shape...> pow(ndarray<Float, Shape...> src, Float rhs)
{
    src.map_v([&rhs](auto&& val) {
        using std::pow;
        return pow(val, rhs);
    });
    return src;
}

template <size_t Axis, size_t N = 0, class Float, size_t... Shape, class Func,
          class Picker>
inline void axis_each(const ndarray<Float, Shape...>& src, Func func,
                      Picker picker)
{
    constexpr auto dim = ndarray<Float, Shape...>::dim();
    constexpr auto shape = ndarray<Float, Shape...>::shape();

    static_assert(Axis < dim);

    if constexpr (N < dim) {
        for (size_t i = 0; i < shape.at(N); i++) {
            if constexpr (N == Axis) {
                axis_each<Axis, N + 1>(
                    src, func, [i, picker](auto&&... indices) {
                        return picker(
                            i, std::forward<decltype(indices)>(indices)...);
                    });
            }
            else {
                axis_each<Axis, N + 1>(
                    src,
                    [i, func](auto&&... indices) {
                        return func(
                            i, std::forward<decltype(indices)>(indices)...);
                    },
                    [i, picker](auto&&... indices) {
                        return picker(
                            i, std::forward<decltype(indices)>(indices)...);
                    });
            }
        }
    }
    else {
        func()(picker());
    }
}

namespace detail {
template <size_t Axis, size_t N = 0, class Float, size_t... Shape>
inline auto axis_each_t(const ndarray<Float, Shape...>& src)
{
    constexpr auto dim = ndarray<Float, Shape...>::dim();
    constexpr auto shape = ndarray<Float, Shape...>::shape();

    static_assert(Axis < dim);

    if constexpr (N < dim) {
        if constexpr (N == Axis) {
            return axis_each_t<Axis, N + 1>(src);
        }
        else {
            return tao::seq::concatenate_t<
                tao::seq::integer_sequence<size_t, shape.at(N)>,
                decltype(axis_each_t<Axis, N + 1>(src))>();
        }
    }
    else {
        return tao::seq::make_integer_range<size_t, 0, 0>();
    }
}
}  // namespace detail

template <size_t Axis, class Float, size_t... Shape>
using axis_each_t = typename detail::seq2ndarray<
    Float, decltype(detail::axis_each_t<Axis>(
               std::declval<ndarray<Float, Shape...>>()))>::type;

template <int Axis = -1, class Float, size_t... Shape>
inline auto sum(const ndarray<Float, Shape...>& src)
{
    if constexpr (Axis < 0) {
        Float ret = 0;
        for (size_t i = 0; i < src.size(); i++)
            ret += src.data[i];
        return ret;
    }
    else {
        auto ret = axis_each_t<Axis, Float, Shape...>::zeros();
        axis_each<Axis>(
            src,
            [&ret](auto&&... indices) {
                return
                    [&ret, indices...](auto&& val) { ret(indices...) += val; };
            },
            [&src](auto&&... indices) { return src(indices...); });
        return ret;
    }
}

template <int Axis = -1, class Float, size_t... Shape>
inline auto argmax(const ndarray<Float, Shape...>& src)
{
    if constexpr (Axis < 0) {
        size_t ret = 0;
        for (size_t i = 0; i < src.size(); i++) {
            if (src.data[ret] < src.data[i])
                ret = i;
        }
        return ret;
    }
    else {
        auto arg = axis_each_t<Axis, size_t, Shape...>::full(-1);
        auto max = axis_each_t<Axis, Float, Shape...>::zeros();
        axis_each<Axis>(
            src,
            [&arg, &max](auto&&... indices) {
                return [&arg, &max, indices...](auto&& pair) {
                    auto&& [axis_index, val] = pair;
                    if (max(indices...) < val) {
                        arg(indices...) = axis_index;
                        max(indices...) = val;
                    }
                };
            },
            [&src](auto&&... indices) {
                return std::make_pair(
                    std::get<Axis>(std::make_tuple(indices...)),
                    src(indices...));
            });
        return arg;
    }
}

template <int Axis = -1, class Float, size_t... Shape>
inline auto max(const ndarray<Float, Shape...>& src)
{
    if constexpr (Axis < 0) {
        Float ret = 0;
        for (size_t i = 0; i < src.size(); i++)
            ret = std::max(ret, src.data[i]);
        return ret;
    }
    else {
        auto ret = axis_each_t<Axis, Float, Shape...>::zeros();
        axis_each<Axis>(
            src,
            [&ret](auto&&... indices) {
                return [&ret, indices...](auto&& pair) {
                    auto&& [axis_index, val] = pair;
                    ret(indices...) = std::max(ret(indices...), val);
                };
            },
            [&src](auto&&... indices) {
                return std::make_pair(
                    std::get<Axis>(std::make_tuple(indices...)),
                    src(indices...));
            });
        return ret;
    }
}

namespace detail {
template <size_t N, class Float, size_t... Shape, class Func, class Picker>
inline void reversed_each(const ndarray<Float, Shape...>& src, Func func,
                          Picker picker)
{
    constexpr auto shape = ndarray<Float, Shape...>::shape();

    if constexpr (N > 0) {
        for (size_t i = 0; i < shape.at(N - 1); i++) {
            reversed_each<N - 1>(
                src,
                [i, func](auto&&... indices) {
                    return func(i, std::forward<decltype(indices)>(indices)...);
                },
                [i, picker](auto&&... indices) {
                    return picker(std::forward<decltype(indices)>(indices)...,
                                  i);
                });
        }
    }
    else {
        func()(picker());
    }
}
}  // namespace detail

template <class Float, size_t... Shape>
inline auto transpose(const ndarray<Float, Shape...>& src)
{
    typename detail::seq2ndarray<
        Float, tao::seq::reverse_t<size_t, Shape...>>::type ret;

    detail::reversed_each<sizeof...(Shape)>(
        src,
        [&ret](auto&&... indices) {
            return [&ret, indices...](auto&& val) { ret(indices...) = val; };
        },
        [&src](auto&&... indices) { return src(indices...); });

    return ret;
}

template <class Float, size_t... Shape>
inline auto make_ndarray(std::initializer_list<Float> lst)
{
    return ndarray<Float, Shape...>(std::vector<Float>(lst.begin(), lst.end()));
}

template <class Float, size_t N, size_t M>
using Matrix = ndarray<Float, N, M>;
template <class Float, size_t N>
using Vector = ndarray<Float, N>;

}  // namespace ndarray_hpp

#ifdef NDARRAY_HPP_ENABLE_CBLAS
#include <cblas.h>
namespace ndarray_hpp {
namespace detail {
// C<M, N> = A<M, K> * B<K, N>
void matmul(size_t M, size_t N, size_t K, const float* a, const float* b,
            float* c)
{
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, a, K, b,
                N, 0, c, N);
}

// C<M, N> = A<M, K> * B<K, N>
void matmul(size_t M, size_t N, size_t K, const double* a, const double* b,
            double* c)
{
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, a, K, b,
                N, 0, c, N);
}

}  // namespace detail

template <class Float, size_t N, size_t M, size_t L>
ndarray<Float, N, L> dot(const ndarray<Float, N, M>& lhs,
                         const ndarray<Float, M, L>& rhs)
{
    auto ret = ndarray<Float, N, L>::zeros();
    detail::matmul(N, L, M, lhs.data.data(), rhs.data.data(), ret.data.data());
    return ret;
}
}  // namespace ndarray_hpp
#else
namespace ndarray_hpp {
template <class Float, size_t N, size_t M, size_t L>
ndarray<Float, N, L> dot(const ndarray<Float, N, M>& lhs,
                         const ndarray<Float, M, L>& rhs)
{
    auto ret = ndarray<Float, N, L>::zeros();
    for (size_t i = 0; i < N; i++)
        for (size_t k = 0; k < M; k++)
            for (size_t j = 0; j < L; j++)
                ret(i, j) += lhs(i, k) * rhs(k, j);
    return ret;
}
}  // namespace ndarray_hpp
#endif

#endif
