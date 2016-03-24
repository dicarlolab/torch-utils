require 'torch'
require 'os'

net = {}

--graph library for manipulating network structure (like python's networkx but worse)
require('graph')
--the best nn library in torch, this is an extension of 'nn' by Nando DeFreitas 
require('nngraph')
sha1 = require('sha1')
require('hdf5')
require('mongo')
-- pl is "penlight" which is a utilies library framework for lua 
local config = require('pl.config')   -- read INI config files 
local utils = require('pl.utils')     
local tablex = require('pl.tablex')   

function flatten()
    F = function (x) 
        local shp = x:size()
        local sz = 1
        for i=2,#shp do
	    sz = shp[i] * sz
	end
        x = x:reshape(shp[1], sz)
        return torch.DoubleTensor(x:size()):copy(x)
    end
    return F
end

function typer()
    F = function(x)
        return torch.DoubleTensor(x:size()):copy(x)
    end
    return F	
end

--using "registry" pattern for postprocessor for data loading 
--(in DLDataProvider2, this is unnecessary since postprocessors can be implemented in the dataset object itself)
net.POSTPROCESSOR_REGISTRY = {flatten=flatten, typer=typer}

--data provider equivalent
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
    self.curr_epoch = 1
end


function HDF5DataProvider:setEpochBatch(epoch, batch_num)
    self.curr_epoch = epoch
    self.curr_batch_num = batch_num
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
    if (self.curr_batch_num >= m-1) then
        self.curr_epoch = self.curr_epoch + 1
    end
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

--function for loading a network from INI config file
function net.loadnet(filename)
    --[[
       input 
       	     filename = path of .ini config file
       returns 
       	       N = network
	       rootnames = names of input nodes in proper order, e.g {datain, labelin} 
	       leafnames = names of output nodes in proper order, e.g. {loss1, loss2}
	       G = graph object describing network
	       steplist = not important
	       stepspecs = list of loaded up per-layer specific description in lua table format

    --]]
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


-- loads, initializes, trains, and saves network (to database)
function net.trainSGDMultiObjective(args)
    --set seed
    seed = args['random_seed']
    torch.manualSeed(seed)

    --load net and set epoch/batch
    local load_file = args['load_file']
    local load_query = args['load_query']
    local config_file = args['config_file']
    local netspec, epoch0, batch_num0, N, rootnames, leafnames, netGraph, steplist, stepspecs, prevs
    if load_file then
        local netspec = torch.load(load_file)
	N = netspec['net']
	rootnames = netspec['rootnames']
	leafnames = netspec['leafnames']
	stepspecs = netspec['stepspecs']
	prevs = netspec['prevs']
	epoch0 = netspec['epoch']
	batch_num0 = netspec['batch_num']
    elseif load_query then
        netspec = loadFromDatabase(load_query)
	N = netspec['_saved_state']['net']
	prevs = netspec['_saved_state']['prevs']
	rootnames = netspec['rootnames']
	leafnames = netspec['leafnames']
	stepspecs = netspec['stepspecs']
	epoch0 = netspec['epoch']
	batch_num0 = netspec['batch_num']
    else 
        N, rootnames, leafnames, netGraph, steplist, stepspecs = net.loadnet(config_file)
	netspec = {}
	epoch0 = 0
	batch_num0 = 0
	prevs = {}
    end
    --initialize experiment_data
    if args['experiment_data'] then experiment_data = args['experiment_data'] else experiment_data = netspec['experiment_data'] end

    --initialize data providers
    local dp_args, outputPatterns
    if args['epoch0'] then epoch0 = args['epoch0'] end
    if args['batch_num0'] then batch_num0 = args['batch_num0'] end
    if args['dp_params'] then dp_args = args['dp_params'] else dp_args = netspec['dp_params'] end
    if args['outputPatterns'] then outputPatterns = args['outputPatterns'] else outputPatterns = netspec['outputPatterns'] end
    assert (#outputPatterns == #dp_args)
    local inputPatterns={}, data_length, dp, total_batches
    for dpind=1,#dp_args do
    	local dp_arg = dp_args[dpind]
	assert (#dp_arg['sourcelist'] == #rootnames)
        dp_arg1 = process_dp_arg(dp_arg)
	dp = net.HDF5DataProvider(dp_arg1)
	dp:setEpochBatch(epoch0, batch_num0)
	inputPatterns[dpind] = dp
	if not data_length then data_length = dp.data_length end
	if not total_batches then total_batches = dp.total_batches end
	assert (data_length == dp.data_length)
	assert ((#leafnames==1) or (#outputPatterns[dpind] == #leafnames))
    end

    -- loop stepSGDMultiObjective for given number of steps
    local num_batches = args['num_batches']
    local weight_decay, learning_rate_params, momentum_params
    if args['weight_decay'] then weight_decay = args['weight_decay'] else weight_decay = netspec['weight_decay'] end
    if args['learning_rate_params'] then 
        learning_rate_params = args['learning_rate_params']
    else
        learning_rate_params = netspec['learning_rate_params']
    end
    if args['momentum_params'] then 
        momentum_params = args['momentum_params']
    else
        momentum_params = netspec['momentum_params']
    end
    local epoch = epoch0
    local batch_num = batch_num0
    local save, momentum, save_filters, save_args, rec, save_freq
    if args['save_freq'] then
        save_freq = args['save_freq']
    else 
        save_freq = netspec['save_freq']
    end
    if args['write_freq'] then
        write_freq = args['write_freq']
    else 
        write_freq = netspec['write_freq']
    end
    for _stepind=1,num_batches do
    	learning_rate = get_learning_rate(learning_rate_params, epoch, batch_num)
        momentum = get_momentum(momentum_params, epoch, batch_num, learning_rate)
        net.stepSGDMultiObjective(N, inputPatterns, outputPatterns,
				  stepspecs, prevs,
				  momentum, learning_rate, weight_decay, rootnames)
        epoch = inputPatterns[1].curr_epoch
	batch_num = inputPatterns[1].curr_batch_num
	assert (batch_num == (_stepind + batch_num0) % total_batches)
	-- print performance
	print('e/b', epoch, batch_num)
        -- save in specified way with specified frequency	
	save =  (_stepind % save_freq  == 0) 
	save_filters = (_stepind % (save_freq * write_freq) == 0)
	if save then
	    -- add performance data    
	    save_args = {experiment_data=experiment_data, 
	                epoch=epoch, batch_num=batch_num,
	   	        dp_params=dp_params, outputPatterns=outputPatterns,
			momentum_params=momentum_params, learning_rate_params=learning_rate_params, 
			weight_decay=weight_decay, rootnames=rootnames, leafnames=leafnames, stepspecs=stepspecs,
			save_freq=save_freq, write_freq=write_freq}
	    rec = saveMongoRec(N, prevs, save_args, args['save_host'], args['save_port'], args['db_name'], args['collection_name'], save_filters)
	end
    end
    
end


function process_dp_arg(dp_arg)
    local dp_arg1 = tablex.deepcopy(dp_arg)
    local processor_name, processor_args, processor_factory, processor
    if dp_arg1['postprocess'] then
        for source, pargs in pairs(dp_arg1['postprocess']) do
            processor_name, processor_args = unpack(pargs)
            processor_factory = net.POSTPROCESSOR_REGISTRY[processor_name]
            processor = processor_factory(processor_args)
	    dp_arg1['postprocess'][source] = processor
        end 
    end
    return dp_arg1
end


function loadFromDatabase(query)
    local host = query['host']
    local port = query['port']
    local dbname = query['dbname']
    local colname = query['collection_name'] + '.files'
    local fsname = query['collection_name']
    local query = tablex.deepcopy(query['query'])
    query["saved_filters"] = true 
    local qspec = {query=query, ordered={timestamp=-1}}
    if not db then
        db = mongo.Connection.New()
        db:connect(host .. ':' .. port)
	fs = mongo.GridFS.New(db, dbname, fsname)
    end
    local r = db:query(dbname .. '.' .. colname, qspec):next()
    local f = fs:find_file({_id=r["_id"]})
    local fstr = ''
    for cind=1,f:num_chunks() do
        fstr = fstr .. f:chunk(cind-1)
    end
    r["_saved_state"] = torch.deserialize(fstr, "ascii")
end

function saveMongoRec(N, prevs, save_args, host, port, dbname, fsname, save_filters)
    local now = os.time()
    save_args = tablex.deepcopy(save_args)
    save_args['timestamp'] = now
    if not db then
        db = mongo.Connection.New()
        db:connect(host .. ':' .. port)
	fs = mongo.GridFS.New(db, dbname, fsname)
    end
    local fsbuilder
    if not fsbuilder then
        fsbuilder = mongo.GridFileBuilder.New(fs)
    end
    print('Serializing model and previous diffs ... ')
    if save_filters then
        fsbuilder:append(torch.serialize({net=N, prevs=prevs}, "ascii"))
	save_args['saved_filters'] = true
    else
	fsbuilder:append(torch.serialize(nil, "ascii"))
    end
    local sastr = torch.serialize(save_args, "ascii")
    print('Creating model filename ... ')
    local fname = sha1(sastr)
    print('... ' .. fname)
    print('building file... ')
    fsbuilder:build(fname)
    local update = {}
    update['$set'] = save_args
    print('updating record')
    db:update(dbname .. '.' .. fsname .. '.files', {filename=fname}, update, false, false)
    print('done saving')
        
end

function get_learning_rate(lr_params, epoch, batch_num)
    return lr_params['base_learning_rate']
end

function get_momentum(momentum_params, epoch, batch_num, learning_rate)
    if momentum_params['base_momentum'] then
        return momentum_params['base_momentum'] 
    else
	return 0
    end
end

function net.stepSGDMultiObjective(N, inputPatterns, outputPatterns, 
	           		    stepspecs, prevs, momentum, learning_rate, 
				    weight_decay)
    --inputPatterns = table of data providers returning object suitable for network intput
    --outputPatterns = table of outputGrad (tables of tensors or single tensors)
    local sourcelist
    for j=1,#inputPatterns do
        if not prevs[j] then prevs[j] = {} end
	inputs1 = inputPatterns[j]:getNextBatch()
	sourcelist = inputPatterns[j].sourcelist
	for iind=1,#sourcelist do
	    inputs[iind] = inputs1[sourcelist[iind]]
	end
	outputGrads = outputPatterns[j]
	net.sgdstep(N, inputs, outputGrads, stepspecs, prevs[j], momentum, learning_rate, weight_decay)
    end
end


function net.sgdstep(N, inputs, outputGrads, stepspecs, prevs, momentum, learning_rate, weight_decay)
    local t0 = os.clock()
    print(N:forward(inputs))
    N:zeroGradParameters()
    N:backward(inputs, outputGrads)
    local t1 = os.clock()
    --print('t1', t1-t0)
    local nodes = N.fg.nodes
    local node, module, spec, lr, wd, lrmult, wdmult, diff, change
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