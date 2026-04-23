#ifndef ROMEO_PRIORITY_QUEUE_H
#define ROMEO_PRIORITY_QUEUE_H

#include <cstddef>
#include <vector>

namespace romeo {

// Bucket priority queue — port of ROMEO.jl src/priorityqueue.jl.
//
// Weights are integers in [1, nbins]. `enqueue` is O(1); `dequeue` is amortized
// O(1). Dequeues pop from the back of each bin (matches Julia's `pop!`) so that
// the tie-breaking order is last-in-first-out within a bin — a reproducibility
// concern that matters when two edges share the same weight.
//
// Uses 1-indexed bin storage (slot 0 is unused) to mirror Julia's 1-based
// `content[weight]` access without translation at every call site.
template <typename T>
class BucketQueue {
   public:
    explicit BucketQueue(int nbins) : nbins_(nbins), min_(nbins + 1), content_(static_cast<std::size_t>(nbins) + 1) {}

    bool empty() const noexcept { return min_ > nbins_; }

    // Lowest non-empty bin, or nbins_+1 if the queue is empty.
    int min_bin() const noexcept { return min_; }

    void enqueue(T item, int weight) {
        content_[static_cast<std::size_t>(weight)].push_back(item);
        if (weight < min_) min_ = weight;
    }

    // Undefined behavior if empty().
    T dequeue() {
        auto& bin = content_[static_cast<std::size_t>(min_)];
        T item = bin.back();
        bin.pop_back();
        while (min_ <= nbins_ && content_[static_cast<std::size_t>(min_)].empty()) ++min_;
        return item;
    }

   private:
    int nbins_;
    int min_;
    std::vector<std::vector<T>> content_;
};

}  // namespace romeo

#endif
