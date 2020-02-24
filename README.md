# ndarray.hpp

Yet another type-safe N-dim array in C++17.

## Usage

```
#include <iostream>

#include "ndarray.hpp"
using namespace ndarray_hpp;

template <class T, size_t N>
void print_array(const std::array<T, N>& src)
{
    std::cout << "[";
    if (N != 0) {
        std::cout << src.at(0);
        for (int i = 1; i < N; i++)
            std::cout << "," << src.at(i);
    }
    std::cout << "]";
}

int main()
{
    ndarray<int, 8, 1, 6, 1> ary0;
    ndarray<int, 7, 1, 5> ary1;
    print_array((ary0 + ary1).shape());  // Print [8, 7, 6, 5].

    ndarray<int, 8, 4, 3> ary2;
    ndarray<int, 2, 1> ary3;
    // print_array((ary2 + ary3).shape());  // This causes compiling error.
}
```

## License

MIT.
