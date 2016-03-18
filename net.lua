require 'torch'
require 'os'

net = {}

require('graph')
require('nngraph')
require('hdf5')
local config = require('pl.config')
local utils = require('pl.utils')
local tablex = require('pl.tablex')


HDF5DataProvider = torch.class('net.HDF5DataProvider')
function HDF5DataProvider:__init(args)
    hdf5source = args['hdf5source']
    sourcelist = args['sourcelist']
    batch_size = args['batch_size']
    subslice = args['subslice']
    postprocess = args['postprocess']
    self.hdf5source = hdf5source
    self.sourcelist = sourcelist
    self.file = hdf5.open(self.hdf5source, 'r')
    self.subslice = subslice
    self.postprocess = postprocess
    self.data = {}
    self.sizes = {}
    for sourceind=1,#sourcelist do
        local source = sourcelist[sourceind]
    	self.data[source] = self.file:read(source)
    end
    for sourceind=1,#sourcelist do
    	source = sourcelist[sourceind]
        if not self.subslice then 
    	    self.sizes[source] = self.data[source]:dataspaceSize()
	else
	    if not self.subsliceinds then
	        if type(self.subslice) == 'function' then
    	            self.subsliceinds = self.subslice(self.file, self.sourcelist)
		else
		    assert(type(self.subslice) == 'string')
		    self.subsliceinds = self.file:read(self.subslice):all()
		end
		local ssnz = self.subsliceinds:nonzero()
		ssnz = ssnz:reshape(torch.LongStorage({ssnz:size(1)}))
		self.subsliceindsnz = ssnz
	    end
            local sz = self.data[source]:dataspaceSize()
	    if not self._orig_data_length then self._orig_data_length = sz[1] end
	    assert (sz[1] == self._orig_data_length, sz[1], self._orig_data_length)
	    sz[1] = self.subsliceinds:sum()
	    self.sizes[source] = sz
	end
   	if not self.data_length then self.data_length = self.sizes[source][1] end
	assert (self.sizes[source][1] == self.data_length, self.sizes[source][1], self.data_length)   
    end
    self.batch_size = batch_size
    self.total_batches = math.ceil(self.data_length / self.batch_size)
    self.curr_batch_num = 0
end


function HDF5DataProvider:setBatchNum(n)
    self.curr_batch_num = n
end


function HDF5DataProvider:getNextBatch()
    local data = {}
    local cbn = self.curr_batch_num
    local startv = cbn * self.batch_size + 1
    local endv = math.min((cbn + 1) * self.batch_size, self.data_length)
    sourcelist = self.sourcelist
    for sourceind=1,#sourcelist do 
        local source = sourcelist[sourceind]
    	local slice = tablex.deepcopy(self.sizes[source])
	table.remove(slice, 1)
        table.insert(slice, 1, {startv, endv})
	for sind=2,#slice do
	    slice[sind] = {1, slice[sind]}
	end
        data[source] = self:getData(self.data[source], slice)
	if self.postprocess and self.postprocess[source] then
	    data[source] = self.postprocess[source](data[source], self.file)
	end
    end
    self:incrementBatchNum()
    return data
end


function HDF5DataProvider:incrementBatchNum()
    m = self.total_batches
    self.curr_batch_num = (self.curr_batch_num + 1) % m
end


function HDF5DataProvider:getData(dsource, slice)
    if not self.subslice then
        return dsource:partial(unpack(slice))
    else
        local ssind = self.subsliceinds
        local ssindnz = self.subsliceindsnz
        local s1 = ssindnz[slice[1][1]]
	local s2 = ssindnz[slice[1][2]]
	local ssind0 = ssind[{{s1, s2}}]
	local nss = math.ceil((s2 - s1 + 1) / self.batch_size)
	local datalist = {}
	local batch_size = self.batch_size
	for j=1,nss do
       	    local startv = s1 + batch_size * (j-1)
	    local endv = math.min(s1 + batch_size * j - 1, self._orig_data_length)
	    local newslice = tablex.deepcopy(slice)
	    newslice[1] = {startv, endv}
	    local data = dsource:partial(unpack(newslice))
	    startv = batch_size * (j-1) + 1
	    endv = math.min(ssind0:size()[1], batch_size * j)
	    dataind = ssind0[{{batch_size * (j-1) + 1, endv}}]:nonzero()
	    dataind = dataind:reshape(torch.LongStorage({dataind:size()[1]}))
	    datalist[j] = data:index(1, dataind)
	end
        if (nss > 1) then
    	    datalist[#datalist + 1] = 1
	    data = torch.cat(unpack(datalist))
        else
            data = datalist[1]
        end
        return data
    end
end


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


function net.trainSGDMultiObjective(args)
    --set seed
    seed = args['random_seed']
    torch.manualSeed(seed)

    --load net and set epoch/batch
    load_file = args['load_file']
    load_query = args['load_query']
    config_file = args['config_file']
    if load_file then
        netspec = torch.load(load_file)
	net = netspec['net']
	epoch = netspec['epoch']
	batch_num = netspec['batch_num']
    elseif load_query then
        netspec = loadFromDatabase(load_query)
	net = netspec['net']
	epoch = netspec['epoch']
	batch_num = netspec['batch_num']
    else 
	net = net.loadnet(config)
	epoch = 0
	batch_num = 0
    end

    -- initialize data provider to batch_num
    -- dpargs = 
    
    -- loop stepSGDMultiObjective for given number of steps
    -- save in specified way with specified frequency
end


function net.stepSGDMultiObjective(N, inputPatterns, outputPatterns, 
	           		    stepspecs, prevs, momentum, learning_rate, 
				    weight_decay)
    --inputPatterns = table of data providers returning object suitable for network intput
    --outputPatterns = table of outputGrad (tables of tensors or single tensors)
    for j=1,#inputPatterns do
        if not prevs[j] then prevs[j] = {} end
	inputs = inputPatterns[j]: getNextBatch()
	outputGrads = outputPatterns[j]
	net.sgdstep(N, inputs, outputGrads, stepspecs, prevs[j], momentum, learning_rate, weight_decay)
    end
end


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