--require "nn"
require "rnn"

print "\nrun.lua"
local timer = torch.Timer() -- Timer starts counting

print "  Setting up..."
-- According to this article: http://karpathy.github.io/2015/05/21/rnn-effectiveness/
local alphabet = "abcdefghijklmnopqrstuvwxyz0123456789 "

-- Hyper-parameters 
local seqlen = 225      -- max number of letters in a tweet
local featsize = #alphabet     -- length of a representation of a letter based on alphabet
local batchsize = 225     -- Number of input samples processed at once
local rho = 1           -- Backpropagation through time
local hiddensize = 10  -- Number of nodes in each hidden layer
local lr = 1e-5         -- Learning rate
local nIters = 5       -- Number of training iterations

-- Input size
local nInput = featsize
-- Output size
local nOutput = 3

local NUM_FOLD = 10

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

-- Recurrent module
local rnn_module = nn.LSTM(nInput, hiddensize)
local rnn = nn.Sequencer(rnn_module)

-- Output module
local out_module = nn.Sequential()
              :add(nn.Linear(hiddensize, nOutput))
              :add(nn.LogSoftMax())
local out = nn.Sequencer(out_module)

-- Combine two modules
local model = nn.Sequential()
                :add(rnn)
                :add(out)
print(model)

-- Put the model in training mode
rnn:training()  

-- Criterion for the model
--local crit_rnn = nn.SequencerCriterion(nn.ClassNLLCriterion())
--local crit_seq = nn.MSECriterion()
local criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

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
      table.insert(
        x, 
        alphabet_map(string.sub(tweet, i, i))
      )
    end
    -- Extend the tweet to seqlen size so all samples have the same size
    for i = 1, seqlen - #tweet do
        table.insert(x, alphabet_map("!"))  -- String of all 0's
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
    -- Process every character in the tweets and add them to a sample
    for i = 1, #tweet do
      table.insert(
        x, 
        alphabet_map(string.sub(tweet, i, i))
      )
    end
    -- Extend the tweet to seqlen size so all samples have the same size
    for i = 1, seqlen - #tweet do
        table.insert(x, alphabet_map("!"))  -- String of all 0's
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
    if label == "-" then table.insert(oba_training_target[fold], alphabetm_map("1")) end
    if label == "0" then table.insert(oba_training_target[fold], alphabetm_map("2")) end
    if label == "+" then table.insert(oba_training_target[fold], alphabetm_map("3")) end
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
    if label == "-" then table.insert(oba_training_target[fold], alphabetm_map("1")) end
    if label == "0" then table.insert(oba_training_target[fold], alphabetm_map("2")) end
    if label == "+" then table.insert(oba_training_target[fold], alphabetm_map("3")) end
  end
  --]]

end

--[[
-- Training
--]]
print("  Training...")
-- Obama
for fold = 1, 1 do
  
  -- Make a copy of the original model
  local tr_inputs = torch.Tensor(oba_training[fold])
  local tr_targets = torch.Tensor(oba_training_target[fold])
  
  local iteration = 1
  local offset = 1
  batchsize = tr_inputs[1]:size(1)

  while iteration < nIters do
    local inputs, targets = torch.Tensor(batchsize, featsize), 
                            torch.Tensor(batchsize)
    
    for i = 1, batchsize do
      inputs[i] = tr_inputs[offset][i]
      --if i < batchsize then
      --  targets[i] = tr_inputs[offset][i+1]
      --else
        targets[i] = tr_targets[offset]
      --end
    end  
    
    model:zeroGradParameters() 
    model:forget() -- forget all past time-steps

    -- 2. forward sequence through rnn
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
    --]]

    iteration = iteration + 1
    if (offset < tr_inputs:size(1) - 1) then 
      offset = offset + 1 
    else 
      offset = 1 
    end
  end

  --[[
  -- Testing
  --]]
  print("  Testing...")
  model:evaluate()
  local te_inputs = torch.Tensor(oba_testing[fold])
  local te_targets = torch.Tensor(oba_testing_target[fold])
  local result = {}
  result[1] = {0, 0, 0}
  result[2] = {0, 0, 0}
  result[3] = {0, 0, 0}

  for i = 1, te_inputs:size(1) do
    local predictions = model:forward(te_inputs[i])
    local prediction = predictions[batchsize]

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
