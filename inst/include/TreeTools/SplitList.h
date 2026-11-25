#ifndef _TREETOOLS_SPLITLIST_H
#define _TREETOOLS_SPLITLIST_H

#include <Rcpp/Lightest>
#include <stdexcept> /* for errors */
#include <vector>    /* for heap allocation */
#include <algorithm> /* for std::fill */

#include "assert.h" /* for ASSERT */
#include "types.h" /* for int16, int32 */

using splitbit = uint_fast64_t;

#define R_BIN_SIZE int16(8)
#define SL_BIN_SIZE int16(64)
#define SL_MAX_BINS int16(511)

// Keep SL_MAX_TIPS + 2 within int16 range (32767) for downstream packages
#define SL_MAX_TIPS (SL_BIN_SIZE * SL_MAX_BINS) // 32704
#define SL_MAX_SPLITS (SL_MAX_TIPS - 3) 

/* Stack allocation thresholds.
 * Trees with n_splits <= SL_STACK_SPLITS AND n_bins <= SL_STACK_BINS
 * use fast stack arrays; larger trees fall back to heap allocation.
 * Kept at the pre-v1.16 values to avoid bloating SplitList objects.
 */
#define SL_STACK_BINS int16(32)
#define SL_STACK_SPLITS int16(SL_BIN_SIZE * SL_STACK_BINS - 3) // 2045

#define INLASTBIN(n, size) int16((size) - int16((size) - int16((n) % (size))) % (size))
#define INSUBBIN(bin, offset)                                  \
  splitbit(x(split, ((bin) * input_bins_per_bin) + (offset)))
#define INBIN(r_bin, bin) ((INSUBBIN((bin), (r_bin))) << (R_BIN_SIZE * (r_bin)))

namespace TreeTools {

  constexpr int input_bins_per_bin = SL_BIN_SIZE / R_BIN_SIZE;

  template<typename T>
  [[nodiscard]] constexpr T power_of_two(int bit_pos) noexcept {
    static_assert(std::is_unsigned_v<T>, "Use unsigned types for bit operations");
    assert(bit_pos >= 0 && bit_pos < int(sizeof(T) * 8));
    return T(1) << bit_pos;
  }
  
  // Population count -------------------------------------------------------
  //
  // POPCNT belongs to x86-64-v2, not to baseline x86-64: executing it on a
  // CPU without it (e.g. QEMU's default qemu64 model) raises SIGILL (#287).
  // On x86-64 the instruction is therefore used only after a CPUID check.
  //
  // - count_bits(x): drop-in, safe everywhere; one well-predicted branch on a
  //   load-time flag per call.
  // - dispatch_popcount(f): for hot loops. Calls f(PopcountHW{}) or
  //   f(PopcountSW{}) once; a kernel templated on the policy and calling
  //   Pop::count(x) runs exactly as fast as unconditional POPCNT, e.g.
  //
  //     template <typename Pop> double kernel(...) { ... Pop::count(x) ... }
  //     return TreeTools::dispatch_popcount([&](auto pop) {
  //       return kernel<decltype(pop)>(...);
  //     });
  //
  // TREETOOLS_POPCOUNT_DISPATCH lets downstream code test for this API.
#define TREETOOLS_POPCOUNT_DISPATCH 1

  inline int32 popcount_sw(splitbit x) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<int32>(__builtin_popcountll(x));
#else
    int32 count = 0;
    while (x != 0) {
      x &= (x - 1);
      ++count;
    }
    return count;
#endif
  }

#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__) && \
    !defined(__POPCNT__)
  inline bool cpu_has_popcnt() noexcept {
    __builtin_cpu_init();
    return __builtin_cpu_supports("popcnt");
  }

  // Evaluated once, when the shared object that includes this header is
  // loaded. Anything that runs before dynamic initialization reads false and
  // takes the portable path, so the instruction is never reached unchecked.
  inline const bool has_popcnt = cpu_has_popcnt();

  // Executes POPCNT unconditionally: call only where has_popcnt is true.
  // volatile stops the compiler hoisting the instruction above that check.
  inline int32 popcount_hw(splitbit x) noexcept {
    uint64_t result;
    __asm__ __volatile__ ("popcnt %1, %0" : "=r" (result) : "r" (x));
    return static_cast<int32>(result);
  }
#elif defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
  // Not used by R; unchanged from previous versions.
  inline constexpr bool has_popcnt = true;
  inline int32 popcount_hw(splitbit x) noexcept {
    return static_cast<int32>(__popcnt64(x));
  }
#else
  // Non-x86 (the builtin maps to the native instruction), or x86-64 built
  // with -mpopcnt / -march=x86-64-v2 or later (it compiles to POPCNT).
  inline constexpr bool has_popcnt = true;
  inline int32 popcount_hw(splitbit x) noexcept { return popcount_sw(x); }
#endif

  inline int32 count_bits(splitbit x) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    if (__builtin_expect(has_popcnt, 1)) return popcount_hw(x);
#else
    if (has_popcnt) return popcount_hw(x);
#endif
    return popcount_sw(x);
  }

  struct PopcountHW {
    static int32 count(splitbit x) noexcept { return popcount_hw(x); }
  };
  struct PopcountSW {
    static int32 count(splitbit x) noexcept { return popcount_sw(x); }
  };

  template <typename F>
  inline decltype(auto) dispatch_popcount(F&& f) {
    if (has_popcnt) return f(PopcountHW{});
    return f(PopcountSW{});
  }

  // Sums the bits set in each split of a packed state.
  template <typename Pop>
  inline void count_in_split_impl(splitbit** state, int32* in_split,
                                  int32 n_splits, int32 n_bins) {
    for (int32 split = 0; split < n_splits; ++split) {
      int32 n = 0;
      for (int32 bin = 0; bin < n_bins; ++bin) {
        n += Pop::count(state[split][bin]);
      }
      in_split[split] = n;
    }
  }

  inline void count_in_split(splitbit** state, int32* in_split,
                             int32 n_splits, int32 n_bins) {
    dispatch_popcount([&](auto pop) {
      count_in_split_impl<decltype(pop)>(state, in_split, n_splits, n_bins);
    });
  }

  // SplitList(const Rbyte*, int32 n_rows, int32 n_cols) is available.
#define TREETOOLS_SPLITLIST_RAW_CTOR 1

  class SplitList {
  public:
    int32 n_splits;
    int32 n_bins;
    int32* in_split;
    splitbit** state;
  
  private:
    /* STACK STORAGE (Fast path for small trees ≤ SL_STACK_SPLITS splits) */
    int32 stack_in_split[SL_STACK_SPLITS];
    splitbit stack_state[SL_STACK_SPLITS][SL_STACK_BINS];
    splitbit* stack_rows[SL_STACK_SPLITS];
    
    /* HEAP STORAGE (Large trees) */
    std::vector<int32> heap_in_split;
    std::vector<splitbit> heap_data;      
    std::vector<splitbit*> heap_rows;     
    
  public:
    // Packs a column-major raw split matrix, as produced by as.Splits():
    // `x` holds `n_rows` splits by `n_cols` bytes, eight tips per byte.
    // Reads only the memory at `x`, so callers that already hold a pointer
    // (e.g. RAW() of a SEXP) can skip Rcpp::as<RawMatrix>().
    SplitList(const Rbyte* x, const int32 n_rows, const int32 n_cols) {
      init(x, n_rows, n_cols);
    }

    SplitList(const Rcpp::RawMatrix &x) {
      
      const double n_rows = static_cast<double>(x.rows());
      
      /* Check limits */
      if (n_rows > static_cast<double>(std::numeric_limits<int32>::max())) {
        Rcpp::stop("Too many splits (exceeds int32 limit).");                   // #nocov
      }
      
      init(x.begin(), int32(x.rows()), int32(x.cols()));
    }
    
    // Default destructor handles vector cleanup automatically
    ~SplitList() = default;
    
    // Disable copy/move to prevent pointer invalidation issues
    SplitList(const SplitList&) = delete;
    SplitList& operator=(const SplitList&) = delete;

  private:
    void init(const Rbyte* x, const int32 n_rows, const int32 n_input_bins) {
      n_splits = n_rows;
      ASSERT(n_splits >= 0);
      ASSERT(n_input_bins >= 0);
      n_bins = int32(n_input_bins + R_BIN_SIZE - 1) / input_bins_per_bin;
      
      const bool use_heap = (n_splits > SL_STACK_SPLITS) ||
        (n_bins > SL_STACK_BINS);
      
      // Row s of the packed state starts at data + s * stride.
      splitbit* data;
      std::size_t stride;
      
      if (use_heap) {
        heap_in_split.resize(n_splits, 0);
        in_split = heap_in_split.data();
        
        stride = static_cast<std::size_t>(n_bins);
        heap_data.resize(static_cast<std::size_t>(n_splits) * stride);
        data = heap_data.data();
        
        heap_rows.resize(n_splits);
        state = heap_rows.data();
      } else {
        in_split = stack_in_split;
        stride = SL_STACK_BINS;
        data = &stack_state[0][0];
        state = stack_rows;
      }
      for (int32 i = 0; i < n_splits; ++i) {
        state[i] = data + i * stride;
      }
      
      // R stores the matrix column-major, so the eight bytes of each 64-bit
      // bin lie n_splits apart. Read through a raw pointer, with the eight
      // loads of a full bin unrolled; Rcpp's per-element indexing is ~3x
      // slower.
      const std::size_t n = static_cast<std::size_t>(n_splits);
      const int32 full_bins = n_input_bins / input_bins_per_bin;
      
      for (int32 bin = 0; bin < full_bins; ++bin) {
        const Rbyte* col = x + static_cast<std::size_t>(bin) *
          input_bins_per_bin * n;
        splitbit* out = data + bin;
        for (std::size_t split = 0; split < n; ++split) {
          const Rbyte* p = col + split;
          out[split * stride] =
            splitbit(p[0]) |
            (splitbit(p[n]) << 8) |
            (splitbit(p[2 * n]) << 16) |
            (splitbit(p[3 * n]) << 24) |
            (splitbit(p[4 * n]) << 32) |
            (splitbit(p[5 * n]) << 40) |
            (splitbit(p[6 * n]) << 48) |
            (splitbit(p[7 * n]) << 56);
        }
      }
      
      if (full_bins < n_bins) {
        // Final, partly filled bin
        const int32 raggedy_bins = n_input_bins - full_bins * input_bins_per_bin;
        const Rbyte* col = x + static_cast<std::size_t>(full_bins) *
          input_bins_per_bin * n;
        splitbit* out = data + full_bins;
        for (std::size_t split = 0; split < n; ++split) {
          splitbit combined = 0;
          for (int32 input_bin = 0; input_bin < raggedy_bins; ++input_bin) {
            combined |= splitbit(col[split + input_bin * n]) <<
              (R_BIN_SIZE * input_bin);
          }
          out[split * stride] = combined;
        }
      }
      
      count_in_split(state, in_split, n_splits, n_bins);
    }
  };
}

#endif
