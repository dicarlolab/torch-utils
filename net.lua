require 'torch'
require 'os'

net = {}

require('graph')
require('nngraph')
local config = require('pl.config')
local utils = require('pl.utils')

function getnode(G, x)
    nodes = G.nodes
    for i=1,#nodes do
        node = nodes[i]
	if node.data == x then
	   return node
        end
    end
    return graph.Node.new(x)
end

function net.loadnet(filename)
    stepspecs = config.read(filename)
    G = graph.Graph()
    steplist = {}
    for k, v in pairs(stepspecs) do
        nodek = getnode(G, k)
	opname = v['op']
	inlist = {}
	inputs = v['inputs']
	if type(inputs) == 'string' then
            inputs = {inputs}
        end
	if inputs then
            for inpind = 1, #inputs do
	        inp = inputs[inpind]
        	inpnode = getnode(G, inp)
                e = graph.Edge.new(inpnode, nodek)
	        G:add(e)
	    end
	end
    end
    nodes = G:topsort()
    steplist = {}
    for nind=1,#nodes do
    	node = nodes[nind]
        nname = node.data
	v = stepspecs[nname]
	args = v['args']
	opname = v['op']
	if args then
	    if type(args) == 'table' then
     	        op = nn[opname](unpack(args))
	    else
		op = nn[opname](args)
	    end
	else
	    op = nn[opname]()
	end
	initW = v['initW']
        if initW then
            op.weight = torch.mul(op.weight, initW)
        end
	if initB then
            op.bias = torch.add(op.bias, initB)
        end
	inputs = v['inputs']
	if inputs then
    	    if type(inputs) == 'string' then
                inputs = {inputs}
            end
	    inputlist = {}
	    for inpind=1,#inputs do
                inputlist[inpind] = steplist[inputs[inpind]]
            end 
	    step = op(inputlist) 
	else
	    step = op()
	end
	step:annotate{name=nname}
	steplist[nname] = step
    end
    roots = G:roots()
    rootnodes = {}
    rootnames = {}
    for rind=1,#roots do
    	rootnames[#rootnames + 1] = roots[rind].data
        rootnodes[#rootnodes + 1] = steplist[roots[rind].data]
    end
    leaves = G:leaves()
    leafnodes = {}
    leafnames= {}
    for lind=1,#leaves do
    	leafnames[#leafnames + 1] = leaves[lind].data
        leafnodes[#leafnodes + 1] = steplist[leaves[lind].data]
    end
    N = nn.gModule(rootnodes, leafnodes)
    return N, rootnames, leafnames, G, steplist, stepspecs
end


--[[
function net.train(N, nbatches, inputPatternMap, outputPatternMap, stepspecs, prevs, momentum, learning_rate, weight_decay, save_frequency)
    
end
--]]

function net.sgdstep(N, inputs, outputGrads, stepspecs, prevs, momentum, learning_rate, weight_decay)
    t0 = os.clock()
    N:forward(inputs)
    N:zeroGradParameters()
    N:backward(inputs, outputGrads)
    t1 = os.clock()
    --print('t1', t1-t0)
    nodes = N.fg.nodes
    for nind=1,#nodes do
    	node = nodes[nind]
        module = node.data.module
	if module then
            name = node.data.annotations["name"]
	    spec = stepspecs[name]
 	    if module.weight then
	        lr = learning_rate
		lrmult = spec['LRmultW']
		if lrmult then
		    lr = lr * lrmult
		end
		wd = weight_decay
		wdmult = spec['WDmultW']
		if wdmult then
		   wd = wd * wdmult
		end
		diff = module.gradWeight
	        change = - diff * lr - module.weight * lr * wd
	        prev_change = prevs[name .. '_weight']
		if prev_change then
                    change = prev_change * momentum + change
		end
    	        module.weight = change + module.weight
                prevs[name .. '_weight'] = change
	    end
 	    if module.bias then
	        lr = learning_rate
		lrmult = spec['LRmultB']
		if lrmult then
		    lr = lr * lrmult
		end
		wd = weight_decay
		wdmult = spec['WDmultB']
		if wdmult then
		   wd = wd * wdmult
		end  
		diff = module.gradBias
	        change = - diff * lr - module.bias * lr * wd
	        prev_change = prevs[name .. '_bias']
		if prev_change then
                    change = prev_change * momentum + change
		end
    	        module.bias = change + module.bias
                prevs[name .. '_bias'] = change
	    end

	end
    end
    t2 = os.clock()
    --print('t2', t2 - t1)
end


return net