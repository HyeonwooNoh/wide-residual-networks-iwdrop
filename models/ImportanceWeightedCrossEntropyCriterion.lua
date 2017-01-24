local ImportanceWeightedCrossEntropyCriterion, Criterion =
	torch.class('nn.ImportanceWeightedCrossEntropyCriterion', 'nn.Criterion')

function ImportanceWeightedCrossEntropyCriterion:__init(num_samples, num_output,
																		  batch_size, weights,
																		  sizeAverage)
	Criterion.__init(self)
	self.num_samples = num_samples
	self.num_output = num_output
	self.batch_size = batch_size
	self.softmax = nn.SoftMax()
	self.cross_entropy = nn.CrossEntropyCriterion(weights, sizeAverage)
	self.onehot_generator = nn.LookupTable(num_output, num_output)
	self.onehot_generator.weight:eye(num_output, num_output)
end

function ImportanceWeightedCrossEntropyCriterion:updateOutput(input, target)
	self.cross_entropy:updateOutput(input, target)
	self.output = self.cross_entropy.output
	return self.output
end
	
function ImportanceWeightedCrossEntropyCriterion:updateGradInput(input, target)
	-- Compute Importance Weight
	local probs = self.softmax:updateOutput(input)
	local onehot_target = self.onehot_generator:updateOutput(target)
	local correct_probs = probs:cmul(onehot_target):sum(2)
	local normalization_term = correct_probs
		:reshape(self.num_samples, self.batch_size):t()
		:sum(2):repeatTensor(self.num_samples, 1)
	local eps = 1e-12
	-- we multiply num_samples to the importance weight as the cross entropy
	-- criterion have already divided the loss with the effective_batch_size
	-- (batch_size * num_samples)
	-- ref:
	-- https://github.com/torch/nn/blob/master/doc/criterion.md#crossentropycriterion
	local importance_weight = correct_probs:cdiv(normalization_term + eps)
		:mul(self.num_samples)

	-- Back-propagation
	self.cross_entropy:updateGradInput(input, target)
	local cross_entropy_gradInput = self.cross_entropy.gradInput
	local importance_weighted_gradInput = cross_entropy_gradInput
		:cmul(importance_weight:repeatTensor(1, self.num_output))
	self.gradInput:view(importance_weighted_gradInput, input:size())
	return self.gradInput
end

return nn.ImportanceWeightedCrossEntropyCriterion
