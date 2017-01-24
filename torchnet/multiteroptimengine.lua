--[[
   Copyright (c) 2016-present, Facebook, Inc.
   All rights reserved.

   This source code is licensed under the BSD-style license found in the
   LICENSE file in the root directory of this source tree. An additional grant
   of patent rights can be found in the PATENTS file in the same directory.

	This code is modified from
	'https://github.com/torchnet/torchnet/blob/master/engine/optimengine.lua'
	by HyeonwooNoh 'https://github.com/HyeonwooNoh'
]]--

local tnt = require 'torchnet.env'
local argcheck = require 'argcheck'
local doc = require 'argcheck.doc'

doc[[

### tnt.MultIterOptimEngine

The `MultIterOptimEngine` module wraps the optimization functions from
https://github.com/torch/optim. At the start of training, the engine will call
`getParameters` on the provided network.

The `train` method requires the following parameters in addition to the
`SGDEngine.train` parameters:

  * `optimMethod` the optimization function (e.g `optim.sgd`)
  * `config` a table with configuration parameters for the optimizer

Example:
```lua
  local engine = tnt.MultIterOptimEngine()
  engine:train{
     network = model,
     criterion = criterion,
     iterator = iterator,
     optimMethod = optim.sgd,
     config = {
        learningRate = 0.1,
        momentum = 0.9,
     },
  }
```
]]

require 'nn'

local MultIterOptimEngine, SGDEngine = torch.class('tnt.MultIterOptimEngine', 'tnt.SGDEngine', tnt)

MultIterOptimEngine.__init = argcheck{
   {name="self", type="tnt.MultIterOptimEngine"},
   call =
      function(self)
         SGDEngine.__init(self)
      end
}

MultIterOptimEngine.train = argcheck{
   {name="self", type="tnt.MultIterOptimEngine"},
   {name="network", type="nn.Module"},
   {name="criterion", type="nn.Criterion"},
   {name="iterator", type="tnt.DatasetIterator"},
   {name="maxepoch", type="number", default=1000},
   {name="optimMethod", type="function"},
   {name="config", type="table", opt=true},
   {name="optimState", type="table", opt=true},
   {name="paramFun", type="function", opt=true},
	{name="numIter", type="number", default=1},
   call =
      function(self, network, criterion, iterator, maxepoch, optimMethod,
                config, optimState, paramFun, numIter)
         local state = {
            network = network,
            criterion = criterion,
            iterator = iterator,
            maxepoch = maxepoch,
            optimMethod = optimMethod,
            sample = {},
            config = config or {},
            optim = optimState or {},
            epoch = 0, -- epoch done so far
            t = 0, -- samples seen so far
            training = true,
				it = 1,
				numIter = numIter or 1,
         }

         if paramFun then
             state.params, state.gradParams = paramFun()
         else
             state.params, state.gradParams = state.network:getParameters()
         end

         local function feval()
            return state.criterion.output, state.gradParams
         end

         self.hooks("onStart", state)
         while state.epoch < state.maxepoch do
            state.network:training()

            self.hooks("onStartEpoch", state)
				state.it = 1
            for sample in state.iterator() do
               state.sample = sample
               self.hooks("onSample", state)

               state.network:forward(sample.input)
               self.hooks("onForward", state)
               state.criterion:forward(state.network.output, sample.target)
               self.hooks("onForwardCriterion", state)
					
					if state.i == 1 then
						state.network:zeroGradParameters()
						if state.criterion.zeroGradParameters then
							state.criterion:zeroGradParameters()
						end
					end

               state.criterion:backward(state.network.output, sample.target)
               self.hooks("onBackwardCriterion", state)
               state.network:backward(sample.input, state.criterion.gradInput:div(state.numIter))
               self.hooks("onBackward", state)
	
					if state.it == state.numIter then
						state.optimMethod(feval, state.params, state.config, state.optim)
					end
					if state.i == state.numIter then
						state.it = 1
					else
						state.it = state.it + 1
					end
               state.t = state.t + 1
               self.hooks("onUpdate", state)
            end
            state.epoch = state.epoch + 1
            self.hooks("onEndEpoch", state)
         end
         self.hooks("onEnd", state)
      end
}
