#include <Rcpp/Lightest>
#include "../inst/include/TreeTools/SplitList.h"
using namespace Rcpp;

static List splitlist_unpack(const TreeTools::SplitList& sl) {
  const int32 n_bytes = sl.n_bins * 8;
  RawMatrix state(sl.n_splits, n_bytes);
  IntegerVector in_split(sl.n_splits);
  for (int32 s = 0; s < sl.n_splits; ++s) {
    in_split[s] = sl.in_split[s];
    for (int32 b = 0; b < n_bytes; ++b) {
      state(s, b) = Rbyte((sl.state[s][b / 8] >> (8 * (b % 8))) & 0xff);
    }
  }
  return List::create(Named("state") = state, Named("in_split") = in_split);
}

// Test helper: build a SplitList and expose its packed contents, unpacked
// back to one byte per eight tips, so tests can compare them to the input.
// [[Rcpp::export]]
List splitlist_contents(const RawMatrix x, const LogicalVector raw_ptr) {
  if (raw_ptr[0]) {
    const TreeTools::SplitList sl(x.begin(), int32(x.nrow()), int32(x.ncol()));
    return splitlist_unpack(sl);
  }
  const TreeTools::SplitList sl(x);
  return splitlist_unpack(sl);
}
