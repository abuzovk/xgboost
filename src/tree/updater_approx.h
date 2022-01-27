/*!
 * Copyright 2021 XGBoost contributors
 *
 * \brief Implementation for the approx tree method.
 */
#ifndef XGBOOST_TREE_UPDATER_APPROX_H_
#define XGBOOST_TREE_UPDATER_APPROX_H_

#include <limits>
#include <utility>
#include <vector>
#include <algorithm>

#include "../common/partition_builder.h"
#include "../common/random.h"
#include "constraints.h"
#include "driver.h"
#include "hist/evaluate_splits.h"
#include "hist/expand_entry.h"
#include "hist/param.h"
#include "param.h"
#include "xgboost/json.h"
#include "xgboost/tree_updater.h"

namespace xgboost {
namespace tree {
class ApproxRowPartitioner {
  using BinIdxType = uint8_t;
  const bool is_loss_guided = false;

  const GHistIndexMatrix* p_gmat_;
  common::OptPartitionBuilder opt_partition_builder_;
  std::vector<uint16_t> node_ids_;
  common::ColumnMatrix column_matrix_;

 public:
  bst_row_t base_rowid = 0;

  static auto SearchCutValue(bst_row_t ridx, bst_feature_t fidx, GHistIndexMatrix const &index,
                             std::vector<uint32_t> const &cut_ptrs,
                             std::vector<float> const &cut_values) {
    int32_t gidx = -1;
    auto const &row_ptr = index.row_ptr;
    auto get_rid = [&](size_t ridx) { return row_ptr[ridx - index.base_rowid]; };

    if (index.IsDense()) {
      gidx = index.index[get_rid(ridx) + fidx];
    } else {
      auto begin = get_rid(ridx);
      auto end = get_rid(ridx + 1);
      auto f_begin = cut_ptrs[fidx];
      auto f_end = cut_ptrs[fidx + 1];
      gidx = common::BinarySearchBin(begin, end, index.index, f_begin, f_end);
    }
    if (gidx == -1) {
      return std::numeric_limits<float>::quiet_NaN();
    }
    return cut_values[gidx];
  }

 public:
  void UpdatePosition(GenericParameter const *ctx,
                      std::vector<CPUExpandEntry> const &candidates,
                      RegTree const *p_tree) {
    size_t n_nodes = candidates.size();
    const GHistIndexMatrix & index = *p_gmat_;

    auto const &cut_values = index.cut.Values();
    auto const &cut_ptrs = index.cut.Ptrs();

    const size_t depth_begin = opt_partition_builder_.DepthBegin({},
                                                               p_tree, is_loss_guided);
    const size_t depth_size = opt_partition_builder_.DepthSize(index, {},
                                                             p_tree, is_loss_guided);
    auto node_ptr = p_tree->GetCategoriesMatrix().node_ptr;
    auto categories = p_tree->GetCategoriesMatrix().categories;

  #pragma omp parallel num_threads(ctx->Threads())
    {
      size_t tid = omp_get_thread_num();
      size_t chunck_size = common::GetBlockSize(depth_size, ctx->Threads());
      size_t begin = chunck_size * tid;
      size_t end = std::min(begin + chunck_size, depth_size);
      begin += depth_begin;
      end += depth_begin;
      opt_partition_builder_.PartitionRange(begin, end, node_ids_.data(), [&](size_t row_id) {
          size_t node_in_set = node_ids_.data()[row_id];
          auto candidate = candidates[node_in_set];
          auto is_cat = candidate.split.is_cat;
          const int32_t nid = candidate.nid;
          auto fidx = candidate.split.SplitIndex();
          auto cut_value = SearchCutValue(row_id, fidx, index, cut_ptrs, cut_values);
          if (std::isnan(cut_value)) {
            return candidate.split.DefaultLeft();
          }
          bst_node_t nidx = candidate.nid;
          auto segment = node_ptr[nidx];
          auto node_cats = categories.subspan(segment.beg, segment.size);
          bool go_left = true;
          if (is_cat) {
            go_left = common::Decision(node_cats, common::AsCat(cut_value));
          } else {
            go_left = cut_value <= candidate.split.split_value;
          }
          return go_left;
        });
    }
    /*Calculate threads work: UpdateRowBuffer, UpdateThreadsWork*/
  }

  auto const &GetNodeAssignments() const { return node_ids_; }

  auto const &GetThreadTasks(const size_t tid) const {
    return opt_partition_builder_.GetSlices(tid);
  }

  ApproxRowPartitioner() = default;
  explicit ApproxRowPartitioner(GenericParameter const *ctx,
                                GHistIndexMatrix const &gmat,
                                const RegTree* p_tree_local,
                                size_t max_depth,
                                bool is_lossguide = false,
                                size_t sparse_threshold = 1): p_gmat_(&gmat) {
    column_matrix_.Init(gmat, 1);
    switch (column_matrix_.GetTypeSize()) {
      case common::kUint8BinsTypeSize:
        opt_partition_builder_.Init<uint8_t>(gmat, column_matrix_, p_tree_local,
                                                ctx->Threads(), max_depth,
                                                is_lossguide);
        break;
      case common::kUint16BinsTypeSize:
        CHECK(false);  // temporal
        opt_partition_builder_.Init<uint16_t>(gmat, column_matrix_, p_tree_local,
                                                ctx->Threads(), max_depth,
                                                is_lossguide);
        break;
      case common::kUint32BinsTypeSize:
        CHECK(false);  // temporal
        opt_partition_builder_.Init<uint32_t>(gmat, column_matrix_, p_tree_local,
                                                ctx->Threads(), max_depth,
                                                is_lossguide);
        break;
      default:
        CHECK(false);  // no default behavior
    }
    node_ids_.resize(gmat.row_ptr.size() - 1, 0);
  }
};
}  // namespace tree
}  // namespace xgboost
#endif  // XGBOOST_TREE_UPDATER_APPROX_H_
