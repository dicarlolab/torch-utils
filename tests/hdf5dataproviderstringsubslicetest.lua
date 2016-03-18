require("../net")

dp = net.HDF5DataProvider("./test.h5", {"data", "labels"}, 10, 'subslice')

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
