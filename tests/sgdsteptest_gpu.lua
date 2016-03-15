require('torch')
require('../net')
require('cunn')
require('cutorch')
--require('cudnn')

torch.manualSeed(0)

N, roots, leaves, G, steplist, stepspecs = net.loadnet('testnet2.ini')
N:cuda()

v = torch.randn(256, 5)
w1 = torch.randn(256, 10)

vg = torch.CudaTensor()
vg:resize(v:size()):copy(v)
w1g = torch.CudaTensor()
w1g:resize(w1:size()):copy(w1)

out = torch.ones(1)
outg = torch.CudaTensor()
outg:resize(out:size()):copy(out)

print(N:forward({vg, w1g}))
prevs = {}
for i=1,100 do
    net.sgdstep(N, {vg, w1g}, outg, stepspecs, prevs, 0, .05, 0.00005)
end
print(N:forward({vg, w1g}))

