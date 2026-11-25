test_that("SplitList packs raw split matrices exactly", {
  Check <- function(x) {
    nBytes <- 8 * ceiling(ncol(x) / 8)
    expected <- matrix(as.raw(0), nrow(x), nBytes)
    expected[, seq_len(ncol(x))] <- x
    bits <- vapply(seq_len(nrow(x)), function(i) {
      sum(as.integer(rawToBits(x[i, ])))
    }, integer(1))
    for (rawPtr in c(FALSE, TRUE)) {
      sl <- splitlist_contents(x, rawPtr)
      expect_identical(sl[["state"]], expected)
      expect_identical(sl[["in_split"]], bits)
    }
  }
  set.seed(0)
  RandomRaw <- function(nRow, nCol) {
    matrix(as.raw(sample.int(256, nRow * nCol, replace = TRUE) - 1L),
           nRow, nCol)
  }
  # Column counts either side of the 8-byte bin boundary
  for (nCol in c(1, 7, 8, 9, 15, 16, 17, 23)) {
    Check(RandomRaw(13, nCol))
  }
  Check(RandomRaw(1, 3))
  Check(matrix(as.raw(0), 0, 5))   # no splits
  Check(matrix(as.raw(255), 4, 8)) # all bits set: top byte shifted fully
  Check(RandomRaw(2046, 3))        # too many splits for stack: heap
  Check(RandomRaw(3, 33 * 8 + 1))  # too many bins for stack: heap
})

test_that("SplitList counts real tree splits", {
  sp <- as.Splits(as.phylo(1:5, 183))
  for (s in sp) {
    expect_identical(splitlist_contents(s, TRUE)[["in_split"]],
                     unname(TipsInSplits(s)))
  }
})
