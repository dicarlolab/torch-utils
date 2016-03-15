require('torch')
require('../net')

torch.manualSeed(0)

N, roots, leaves, G, steplist, stepspecs = net.loadnet('testnet2.ini')

v = torch.randn(5, 5)
w1 = torch.randn(5, 10)

print(N:forward({v, w1}))
prevs = {}
for i=1,10000 do
    net.sgdstep(N, {v, w1}, torch.ones(1), stepspecs, prevs, 0, 1, 0.00005)
end
print(N:forward({v, w1}))

