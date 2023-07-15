import torch

class queue_with_pro:
    def __init__(self, args):
        self.K = 480
        self.p1 = -1.0 * torch.ones(self.K, 128).cuda()
        self.p2 = -1.0 * torch.ones(self.K, 128).cuda()
        self.z1 = -1.0 * torch.ones(self.K, 128).cuda()
        self.z2 = -1.0 * torch.ones(self.K, 128).cuda()
        self.outputs =  -1.0 * torch.ones(self.K, args.nb_classes).cuda()

        self.ptr = 0

    @property
    def is_full(self):
        return self.ptr == self.K or self.ptr == 0

    def get(self):
        if self.is_full:
            return self.p1, self.p2, self.z1, self.z2, self.outputs

        else:
            return self.p1[:self.ptr], self.p2[:self.ptr], self.z1[:self.ptr], self.z2[:self.ptr], self.outputs[:self.ptr]


    def enqueue_dequeue(self, p1, p2, z1, z2, outputs):
        q_size = len(outputs)

        # print("self.ptr="+str(self.ptr)+" q_size="+str(len(outputs)))

        if self.ptr + q_size > self.K:
            self.p1[-q_size:] = p1
            self.p2[-q_size:] = p2
            self.z1[-q_size:] = z1
            self.z2[-q_size:] = z2
            self.outputs[-q_size:] = outputs
            self.ptr = 0
        else:
            self.p1[self.ptr: self.ptr + q_size] = p1
            self.p2[self.ptr: self.ptr + q_size] = p2
            self.z1[self.ptr: self.ptr + q_size] = z1
            self.z2[self.ptr: self.ptr + q_size] = z2
            self.outputs[self.ptr: self.ptr + q_size] = outputs
            self.ptr += q_size