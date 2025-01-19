""" change con-loss  to calculate NDCG improvement
change trainer class in trainer file to the following :
"""


def train(self, epochs, log_interval=100):
    max_results = defaultdict(float)
    max_results['Loss'] = -np.inf
    max_epochs = defaultdict(int)
    bad_counter = 0
    t = time.time()
    total_loss, total_con_loss, mean_loss, mean_con_loss = 0, 0, 0, 0
    initial_beta = self.beta + 0.8  # ערך התחלתי ל-Con-Loss

    for epoch in range(epochs):
        self.model.train()

        # Dynamic Beta for Con-Loss
        dynamic_beta = initial_beta * (1 - epoch / epochs)

        for batch in self.train_loader:
            batch = prepare_batch(batch)

            self.optimizer.zero_grad()

            # Forward pass
            scores, con_loss = self.model(batch, cl=self.contrastive)

            # Main Loss
            main_loss = self.loss_function(scores, batch['targets'])

            # Combined Loss
            if self.contrastive:
                combined_loss = main_loss + dynamic_beta * self.custom_con_loss(scores, batch['targets'])
                combined_loss.backward()
            else:
                main_loss.backward()

            self.optimizer.step()

            # Logging
            if log_interval:
                mean_loss += main_loss.item() / log_interval
                mean_con_loss += con_loss.mean().item() / log_interval

            total_loss += main_loss.item()
            total_con_loss += con_loss.mean().item()

            if log_interval and self.batch > 0 and self.batch % log_interval == 0:
                print(
                    f'Batch {self.batch}: Loss = {mean_loss:.4f}, Con-Loss = {mean_con_loss:.4f}, '
                    f'Dynamic Beta = {dynamic_beta:.4f}, Time Elapsed = {time.time() - t:.2f}s'
                )
                t = time.time()
                mean_loss, mean_con_loss = 0, 0
            self.batch += 1

        # Evaluate model at the end of the epoch
        curr_results = evaluate(self.model, self.test_loader, Ks=self.Ks)

        if log_interval:
            print(f'\nEpoch {epoch}:')
            print('Loss:\t%.3f' % total_loss)
            print('Con-Loss:\t%.3f' % total_con_loss)
            print_results(curr_results)

        # Check for better results
        any_better_result = False
        for metric in curr_results:
            if curr_results[metric] > max_results[metric]:
                max_results[metric] = curr_results[metric]
                max_epochs[metric] = epoch
                any_better_result = True

        if any_better_result:
            bad_counter = 0
        else:
            bad_counter += 1
            if bad_counter == self.patience:
                break

        # Adjust learning rate
        self.scheduler.step()
        self.epoch += 1
        total_loss = 0.0
        total_con_loss = 0.0

    # Print the best results
    print('\nBest results')
    print_results(max_results, max_epochs)
    return max_results


def custom_con_loss(self, scores, targets):
    """
    Compute the contrastive loss based on NDCG (Normalized Discounted Cumulative Gain).

    :param scores: Predicted scores for each item (batch_size, num_items).
    :param targets: Ground truth labels (batch_size).
    :return: NDCG-based loss.
    """
    batch_size = scores.size(0)

    # Create relevance scores: 1 for the target item, 0 for others
    relevance = torch.zeros_like(scores)
    relevance[torch.arange(batch_size), targets] = 1.0

    # Compute DCG: Discounted Cumulative Gain
    _, sorted_indices = torch.sort(scores, descending=True, dim=1)
    sorted_relevance = torch.gather(relevance, dim=1, index=sorted_indices)
    discount_factors = torch.log2(torch.arange(2, sorted_relevance.size(1) + 2).float().to(scores.device))
    dcg = (sorted_relevance / discount_factors).sum(dim=1)

    # Compute IDCG: Ideal DCG
    ideal_relevance = torch.sort(relevance, descending=True, dim=1)[0]
    ideal_discount_factors = discount_factors[:ideal_relevance.size(1)]
    idcg = (ideal_relevance / ideal_discount_factors).sum(dim=1)

    # Compute NDCG
    ndcg = dcg / (idcg + 1e-10)  # Add a small epsilon to avoid division by zero

    # Loss is 1 - NDCG, averaged across the batch
    ndcg_loss = 1 - ndcg
    return ndcg_loss.mean()