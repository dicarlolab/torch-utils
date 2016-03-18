require("../net")

func = function(x, y) return torch.ge(x[y[2]]:all(), 5) end
dp = net.HDF5DataProvider("./test.h5", {"data", "labels"}, 10, func)

assert(dp.total_batches == 18)
assert(dp.data_length == 176)
assert(dp.curr_batch_num == 0)
b = dp:getNextBatch()
assert(dp.curr_batch_num == 1)
assert(b["data"]:size()[1] == 10)
assert(b["data"]:size()[2] == 3)

for i=1,17 do
    assert (dp.curr_batch_num == i)
    b = dp:getNextBatch()
end
assert (dp.curr_batch_num == 0)
