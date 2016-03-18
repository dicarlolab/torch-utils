require("../net")

dp = net.HDF5DataProvider({hdf5source="./test.h5", sourcelist={"data", "labels"}, batch_size=10})
assert(dp.total_batches == 26)
assert(dp.data_length == 256)
assert(dp.curr_batch_num == 0)
b = dp:getNextBatch()
assert(dp.curr_batch_num == 1)
assert(b["data"]:size()[1] == 10)
assert(b["data"]:size()[2] == 3)

for i=1,25 do
    assert (dp.curr_batch_num == i)
    b = dp:getNextBatch()
end
assert (dp.curr_batch_num == 0)
