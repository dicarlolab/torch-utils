require("../net")

dp = net.HDF5DataProvider({hdf5source="./test.h5", 
			   sourcelist={"data", "labels"}, 
			   batch_size=10, 
			   subslice='subslice'})

assert(dp.total_batches == 13)
assert(dp.data_length == 129)
assert(dp.curr_batch_num == 0)
b = dp:getNextBatch()
assert(dp.curr_batch_num == 1)
assert(b["data"]:size()[1] == 10)
assert(b["data"]:size()[2] == 3)

for i=1,12 do
    assert (dp.curr_batch_num == i)
    b = dp:getNextBatch()
end
assert (dp.curr_batch_num == 0)
