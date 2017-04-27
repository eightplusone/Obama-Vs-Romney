--require "nn"
require "rnn"

print "\nrun.lua"
timer = torch.Timer() -- Timer starts counting

print "  Setting up..."
-- According to this paper: http://karpathy.github.io/2015/05/21/rnn-effectiveness/
local alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"

-- Hyper-parameters 
seqlen = 256      -- max number of letters in a tweet
featsize = 36     -- length of a representation of a letter based on alphabet
batchsize = 1     -- Number of input samples processed at once
rho = 1           -- Backpropagation through time
hiddenSize = 512  -- Number of nodes in each hidden layer
lr = 1e-5         -- Learning rate
nIters = 20       -- Number of training iterations

-- Input size
nInput = seqlen * batchsize * featsize
-- Output size
nOutput = 3

function file_exists(name)
  local f = io.open(name,"r")
  if f ~= nil then io.close(f) return true else return false end
end

function alphabet_map(character)
  local output = {}
  for i = 1, #alphabet do
    if string.sub(alphabet, i, i) == character then
      table.insert(output, 1)
    else
      table.insert(output, 0)
    end
  end
  return output
end

-- Build simple recurrent neural network
print "  Creating model..."
local rnn = nn.FastLSTM(nInput, nOutput, rho)
--[[
local rnn = nn.Sequential()
              :add(nn.Linear(nInput, hiddenSize))
              :add(nn.SeqLSTM(hiddenSize, hiddenSize))  -- nn.Sequencer(nn.FastLSTM)
              :add(nn.Linear(hiddenSize, nInput))
              :add(nn.LogSoftMax())
              :add(nn.Linear(nInput, nOutput))
rnn = nn.Sequencer(rnn)
--]]
print(rnn)

-- Put the model in training mode
rnn:training()  

-- Criterion for the model
criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

local NUM_FOLD = 10

-- Check if the txt input files exist
local oba_txt_path = "tmp/oba_processed/"
local rom_txt_path = "tmp/rom_processed/"
if not file_exists(oba_txt_path .. "tr_01.txt") or not file_exists(rom_txt_path .. "tr_01.txt") then
  os.execute("python 1-generate_folds.py")
  os.execute("python 2-process_train_data.py")
end

local oba_training, oba_training_target,
      oba_testing, oba_testing_target,
      rom_training, rom_training_target,
      rom_testing, rom_testing_target 
        = {}, {}, {}, {}, {}, {}, {}, {}

print("  Loading data...")
for fold = 1, 1 do
  print("    Fold # " .. fold)
  oba_training[fold], oba_training_target[fold],
      oba_testing[fold], oba_testing_target[fold],
      rom_training[fold], rom_training_target[fold],
      rom_testing[fold], rom_testing_target[fold] 
        = {}, {}, {}, {}, {}, {}, {}, {}

  --[[
  -- Obama
  --]] 
  print("      Obama")
  local tr_file_name = oba_txt_path
  local te_file_name = oba_txt_path

  if fold < NUM_FOLD then
    tr_file_name = tr_file_name .. "tr_0" .. fold .. ".txt"
  else
    tr_file_name = tr_file_name .. "tr_" .. fold .. ".txt"
  end

  local tr_file = assert(io.open(tr_file_name, "r"))

  while true do
    line = tr_file:read()
    if line == nil then break end

    local label = string.sub(line, 0, 1)
    local tweet = string.sub(line, 7)

    -- Convert string into an array of ASCII codes
    local x = {}
    -- Process every character in the tweets and add them to a sample
    for i = 1, #tweet do
      local converted_string = alphabet_map(string.sub(tweet, i, i))
      for j = 1, #alphabet do
        table.insert(x, converted_string[j])
      end
    end
    -- Extend the tweet to seqlen size so all samples have the same size
    for i = 1, seqlen - #tweet do
      for j = 1, #alphabet do
        table.insert(x, 0)
      end
    end

    -- Insert x into training set
    table.insert(oba_training[fold], x)

    -- Insert values into target set
    if label == "-" then table.insert(oba_training_target[fold], 1) end
    if label == "0" then table.insert(oba_training_target[fold], 2) end
    if label == "+" then table.insert(oba_training_target[fold], 3) end
  end

  if fold < NUM_FOLD then
    te_file_name = te_file_name .. "te_0" .. fold .. ".txt"
  else
    te_file_name = te_file_name .. "te_" .. fold .. ".txt"
  end

  local te_file = assert(io.open(te_file_name, "r"))

  while true do
    line = te_file:read()
    if line == nil then break end

    local label = string.sub(line, 0, 1)
    local tweet = string.sub(line, 7)

    -- Convert string into an array of ASCII codes
    local x = {}
    for i = 1, #tweet do
      l = alphabet_map(string.sub(tweet, i, i))
      for j = 1, #l do
        table.insert(x, l[j])
      end
    end
    for i = 1, seqlen - #tweet do
      for j = 1, #alphabet do
        table.insert(x, 0)
      end
    end

    -- Insert x into training set
    table.insert(oba_testing[fold], x)

    -- Insert values into target set
    if label == "-" then table.insert(oba_testing_target[fold], 1) end
    if label == "0" then table.insert(oba_testing_target[fold], 2) end
    if label == "+" then table.insert(oba_testing_target[fold], 3) end
  end


  --[[
  -- Romney
  --
  print("      Romney")
  tr_file_name = rom_txt_path
  te_file_name = rom_txt_path

  if fold < NUM_FOLD then
    tr_file_name = tr_file_name .. "tr_0" .. fold .. ".txt"
  else
    tr_file_name = tr_file_name .. "tr_" .. fold .. ".txt"
  end

  tr_file = assert(io.open(tr_file_name, "r"))

  while true do
    line = tr_file:read()
    if line == nil then break end

    local label = string.sub(line, 0, 1)
    local tweet = string.sub(line, 7)

    -- Convert string into an array of ASCII codes
    local x = {}
    for i = 1, #tweet do
      l = alphabet_map(string.sub(tweet, i, i))
      for j = 1, #l do
        table.insert(x, l[j])
      end
    end
    for i = 1, seqlen - #tweet do
      for j = 1, #alphabet do
        table.insert(x, 0)
      end
    end

    -- Insert x into training set
    table.insert(rom_training[fold], x)

    -- Insert values into target set
    if label == "-" then table.insert(rom_training_target[fold], 1) end
    if label == "0" then table.insert(rom_training_target[fold], 2) end
    if label == "+" then table.insert(rom_training_target[fold], 3) end
  end

  if fold < NUM_FOLD then
    te_file_name = te_file_name .. "te_0" .. fold .. ".txt"
  else
    te_file_name = te_file_name .. "te_" .. fold .. ".txt"
  end

  local te_file = assert(io.open(te_file_name, "r"))

  while true do
    line = te_file:read()
    if line == nil then break end

    local label = string.sub(line, 0, 1)
    local tweet = string.sub(line, 7)

    -- Convert string into an array of ASCII codes
    local x = {}
    for i = 1, #tweet do
      l = alphabet_map(string.sub(tweet, i, i))
      for j = 1, #l do
        table.insert(x, l[j])
      end
    end
    for i = 1, seqlen - #tweet do
      for j = 1, #alphabet do
        table.insert(x, 0)
      end
    end

    -- Insert x into training set
    table.insert(rom_testing[fold], x)

    -- Insert values into target set
    if label == "-" then table.insert(rom_testing_target[fold], 1) end
    if label == "0" then table.insert(rom_testing_target[fold], 2) end
    if label == "+" then table.insert(rom_testing_target[fold], 3) end
  end
  --]]

end

--[[
-- Training & Testing
--]]
print("  Training & Testing...")
-- Obama
for fold = 1, 1 do
  print("    Fold # " .. fold)
  -- Make a copy of the original model, including the generated weights
  local model = rnn

  --[[
  -- Training
  --]]
  print("      Training...")
  local tr_inputs = torch.Tensor(oba_training[fold])
  local tr_targets = torch.Tensor(oba_training_target[fold])
  
  offsets = {}
  for i = 1, batchsize do
    table.insert(offsets, math.ceil(math.random() * tr_inputs:size(1)))
  end
  offsets = torch.LongTensor(offsets)
  
  local iteration = 1
  while iteration < nIters do
    local inputs, targets = torch.Tensor(batchsize, nInput), torch.Tensor(batchsize, 1)
    
    for step = 1, rho do
      -- 1. create a sequence of rho time-steps
      -- a batch of inputs
      inputs[step] = tr_inputs:index(1, offsets)
      -- incement indices
      offsets:add(1)
      for j = 1, batchsize do
         if offsets[j] > tr_inputs:size(1) then
            offsets[j] = 1
         end
      end
      targets[step] = tr_targets:index(1, offsets)
    end
    
    -- 2. forward sequence through rnn
    model:zeroGradParameters() 
    model:forget() -- forget all past time-steps
    local outputs = model:forward(inputs)
    local err = criterion:forward(outputs, targets)

    --if iteration % 200 == 0 then 
      print(string.format("Iteration %d; err = %f ", iteration, err)) 
    --end

    -- 3. backward sequence through rnn (i.e. backprop through time)
    local gradOutputs = criterion:backward(outputs, targets)
    local gradInputs = model:backward(inputs, gradOutputs)

    -- 4. update
    model:updateParameters(lr)

    iteration = iteration + 1
  end

  --[[
  -- Testing
  --]]
  print("      Testing...")
  local te_inputs = torch.Tensor(oba_testing[fold])
  local te_targets = torch.Tensor(oba_testing_target[fold])
  local result = {}
  result[1] = {0, 0, 0}
  result[2] = {0, 0, 0}
  result[3] = {0, 0, 0}

  for i = 1, te_inputs:size(1) do
    local prediction = model:forward(te_inputs[i])

    local pred = 1
    if (prediction[1] >= prediction[2]) and (prediction[1] >= prediction[3]) then pred = 1 end
    if (prediction[2] >= prediction[1]) and (prediction[2] >= prediction[3]) then pred = 2 end
    if (prediction[3] >= prediction[1]) and (prediction[3] >= prediction[2]) then pred = 3 end

    --print(te_targets[i] .. " " .. pred .. " " .. result[te_targets[i]])
    result[te_targets[i]][pred] = result[te_targets[i]][pred] + 1
  end

  print(result)
end

print('Time elapsed: ' .. timer:time().real .. ' seconds')
