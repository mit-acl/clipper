![banner](https://github.com/mit-acl/clipper/wiki/assets/banner.png)

CLIPPER: A Graph-Theoretic Framework for Robust Data Association
================================================================

Data association is a fundamental problem in robotics and autonomy. CLIPPER provides a framework for robust, pairwise data association and is applicable in a wide variety of problems (e.g., point cloud registration, sensor calibration, place recognition, etc.). By leveraging the notion of *geometric consistency*, a graph is formed and the data association problem is reduced to the [maximum clique problem](https://en.wikipedia.org/wiki/Clique_problem). This NP-hard problem has been studied in many fields, including data association, and solutions techniques are either exact (and not scalable) or approximate (and potentially imprecise). CLIPPER relaxes this problem in a way that (1) **allows guarantees** to be made on the solution of the problem and (2) is applicable to weighted graphs, avoiding the loss of information due to binarization which is common in other data association work. These features allow CLIPPER to achieve high performance, even in the presence of extreme outliers.

This repo provides both MATLAB and C++ implementations of the CLIPPER framework. In addition, Python bindings, Python, C++, and MATLAB examples are included.

## Citation

If you find this code useful in your research, please cite our paper:

- P.C. Lusk, K. Fathian, and J.P. How, "CLIPPER: A Graph-Theoretic Framework for Robust Data Association," arXiv preprint arXiv:2011.10202, 2020. ([**pdf**](https://arxiv.org/pdf/2011.10202.pdf))

```bibtex
@article{lusk2020clipper,
  title={CLIPPER: A Graph-Theoretic Framework for Robust Data Association},
  author={Lusk, Parker C and Fathian, Kaveh and How, Jonathan P},
  journal={arXiv preprint arXiv:2011.10202},
  year={2020}
}
```

## Getting Started

After cloning this repo, please build using `cmake`:

```bash
$ mkdir build
$ cd build
$ cmake ..
$ make
```

Once successful, the C++ tests can be run with `./test/tests`.

### Configuring the Build

The following `cmake` options are available when building CLIPPER:

| Option                  | Description                                                                                                                                                                     | Default |
|-------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|
| `BUILD_BINDINGS_PYTHON` | Uses [`pybind11`](https://github.com/pybind/pybind11) to create Python bindings for CLIPPER                                                                                     | `ON`    |
| `BUILD_BINDINGS_MATLAB` | Attempts to build MEX files which are required for the MATLAB examples. A MATLAB installation is required.                                                                      | `ON`    |
| `BUILD_TESTS`           | Builds C++ tests                                                                                                                                                                | `ON`    |
| `ENABLE_MKL`            | Attempts to use [Intel MKL](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html) (if installed) with Eigen for accelerated linear algebra. | `OFF`   |
| `ENABLE_BLAS`           | Attempts to use a BLAS with Eigen for accelerated linear algebra.                                                                                                               | `OFF`   |

**Note:** The options `ENABLE_MKL` and `ENABLE_BLAS` are mutually exclusive.

These `cmake` options can be set using the syntax `cmake -DENABLE_MKL=ON ..` or using the `ccmake .` command (both from the `build` dir).

### Performance with MKL vs BLAS

On Intel CPUs, MKL should be preferred as it offers superior performance over other general BLAS packages. Also note that on Ubuntu, OpenBLAS (`sudo apt install libopenblas-dev`) provides better performance than the default installed `blas`.

With MKL, we have found an almost 2x improvement in runtime over the MATLAB implementation. On an i9, the C++/MKL implementation can solve problems with 1000 associations in 70 ms.

---

(c) MIT, Ford Motor Company, 2020-2021