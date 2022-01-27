/*!
 * Copyright 2021 XGBoost contributors
 */
#include <gtest/gtest.h>

#include "../../../src/tree/updater_approx.h"
#include "../helpers.h"

namespace xgboost {
namespace tree {
TEST(Approx, Partitioner) {
  size_t n_samples = 1024, n_features = 1, base_rowid = 0;

  auto Xy = RandomDataGenerator{n_samples, n_features, 0}.GenerateDMatrix(true);
  GenericParameter ctx;
  ctx.InitAllowUnknown(Args{});
  std::vector<CPUExpandEntry> candidates{{0, 0, 0.4}};

  for (auto const &page : Xy->GetBatches<GHistIndexMatrix>({GenericParameter::kCpuId, 64})) {
    bst_feature_t split_ind = 0;
    {
      auto min_value = page.cut.MinValues()[split_ind];
      RegTree tree;
      tree.ExpandNode(
          /*nid=*/0, /*split_index=*/0, /*split_value=*/min_value,
          /*default_left=*/true, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
          /*left_sum=*/0.0f,
          /*right_sum=*/0.0f);
      ApproxRowPartitioner partitioner(&ctx, page, &tree, 2);
      candidates.front().split.split_value = min_value;
      candidates.front().split.sindex = 0;
      candidates.front().split.sindex |= (1U << 31);
      partitioner.UpdatePosition(&ctx, candidates, &tree);

      auto const & assignments = partitioner.GetNodeAssignments();
      std::vector<size_t> result(2, 0);
      for (auto node_id : assignments) {
        ++result[node_id];
      }
      ASSERT_EQ(result[0], 0);
      ASSERT_EQ(result[1], n_samples);
    }
    {
      //ApproxRowPartitioner partitioner{n_samples, base_rowid};
      auto ptr = page.cut.Ptrs()[split_ind + 1];
      float split_value = page.cut.Values().at(ptr / 2);
      RegTree tree;
      tree.ExpandNode(
          /*nid=*/RegTree::kRoot, /*split_index=*/split_ind,
          /*split_value=*/split_value,
          /*default_left=*/true, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
          /*left_sum=*/0.0f,
          /*right_sum=*/0.0f);
      ApproxRowPartitioner partitioner(&ctx, page, &tree, 2);
      candidates.front().split.split_value = split_value;
      candidates.front().split.sindex = 0;
      candidates.front().split.sindex |= (1U << 31);
      partitioner.UpdatePosition(&ctx, candidates, &tree);
      auto const & assignments = partitioner.GetNodeAssignments();
      size_t row_id = 0;
      for (auto node_id : assignments) {
        if (node_id == 0) { /* left child */
          auto value = page.cut.Values().at(page.index[row_id++]);
          ASSERT_LE(value, split_value);
        } else {            /* right child */
          auto value = page.cut.Values().at(page.index[row_id++]);
          ASSERT_GT(value, split_value);          
        }
      }
    }
  }
}
}  // namespace tree
}  // namespace xgboost
