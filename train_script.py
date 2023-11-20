import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
import math
import utils
import numpy as np
import os
import pandas as pd
from torch.utils import data
import tqdm
import time
import datetime
from scipy import sparse


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LFM1bDataset(data.Dataset):
    def __init__(
        self,
        root,
        item_mapper,
        user_mapper,
        target=["country"],
        fold_in=True,
        split="train",
        conditioned_on=None,
        upper=-1,
    ):
        super(LFM1bDataset, self).__init__()
        assert os.path.exists(root), "root: {} not found.".format(root)
        self.root = root

        assert split in ["test", "inference", "train", "valid"]
        self.split = split

        out_data_dir = root
        self.target = target
        self.user_mapper = user_mapper
        self.class_list = sorted(
            pd.unique(self.user_mapper[self.target].values.ravel("K")).tolist()
        )
        if self.split == "train":
            self.train_data = pd.read_csv(
                root + "user_interactions_train.csv",
                dtype=np.float64,
                na_filter=False,
                low_memory=False,
            )
        elif self.split == "valid":
            self.vad_data_tr = pd.read_csv(
                root + "user_interactions_validation_tr.csv",
                dtype=np.float64,
                na_filter=False,
                low_memory=False,
            )
            self.vad_data_te = pd.read_csv(
                root + "user_interactions_validation_te.csv",
                dtype=np.float64,
                na_filter=False,
                low_memory=False,
            )
        elif self.split == "test":
            self.test_data_tr = pd.read_csv(
                root + "user_interactions_test_tr.csv",
                dtype=np.float64,
                na_filter=False,
                low_memory=False,
            )
            self.test_data_te = pd.read_csv(
                root + "user_interactions_test_te.csv",
                dtype=np.float64,
                na_filter=False,
                low_memory=False,
            )
        else:
            raise NotImplementedError

        if self.split == "train":
            self.n_users = self.train_data.shape[0]
        elif self.split == "valid":
            self.n_users = self.vad_data_tr.shape[0]
        elif self.split == "test":
            self.n_users = self.test_data_tr.shape[0]
        else:
            raise NotImplementedError

    def __len__(self):
        return self.n_users

    def encode_label(self, label, class_list):
        target = np.zeros(len(class_list))
        for l in label:
            idx = class_list.index(l)
            target[idx] = 1
        return target

    def __getitem__(self, index):
        prof = np.zeros(1)
        if self.split == "train":
            data_tr, data_te = self.train_data.iloc[index].drop("user_id").to_numpy(
                dtype="float32"
            ), np.zeros(1)
            idx_user = self.train_data.at[index, "user_id"]
        elif self.split == "valid":
            # un comment line when vad_data_te is available
            data_tr, data_te = self.vad_data_tr.iloc[index].drop("user_id").to_numpy(
                dtype="float32"
            ), self.vad_data_te.iloc[index].drop("user_id").to_numpy(dtype="float32")
            # data_tr, data_te = self.vad_data_tr.iloc[index].drop('user_id').to_numpy(dtype='float32'), np.zeros(1)
            idx_user = self.vad_data_tr.at[index, "user_id"]
        elif self.split == "test":
            data_tr, data_te = self.test_data_tr.iloc[index].drop("user_id").to_numpy(
                dtype="float32"
            ), self.test_data_te.iloc[index].drop("user_id").to_numpy(dtype="float32")
            idx_user = self.test_data_tr.at[index, "user_id"]

        sensitive = self.user_mapper.loc[self.user_mapper.user_id == idx_user][
            self.target
        ].values[0]
        class_target = self.encode_label(
            label=sensitive,
            class_list=self.class_list,
        )
        return data_tr, data_te, prof, idx_user, class_target


item_mapper = pd.read_csv("./Data/items.csv")
user_mapper = pd.read_csv("./Data/users.csv")


def get_summed_unique_values_count(df, columns):
    # Get the count of unique values for the specified columns
    unique_values_count = df[columns].nunique()

    # Sum the unique values counts across columns
    total_count = unique_values_count.sum()

    return total_count


# Define a deeper neural network architecture
class MultiLabelClassifier(nn.Module):
    def __init__(
        self, input_size=200, hidden_size1=128, hidden_size2=64, num_classes=None
    ):
        super(MultiLabelClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


class Encoder(nn.Module):
    def __init__(self, options, dropout_p=0.5, q_dims=[20108, 600, 200]):
        super(Encoder, self).__init__()
        self.options = options
        self.q_dims = q_dims

        self.dropout = nn.Dropout(p=dropout_p, inplace=False)
        self.linear_1 = nn.Linear(self.q_dims[0], self.q_dims[1], bias=True)
        self.linear_2 = nn.Linear(self.q_dims[1], self.q_dims[2] * 2, bias=True)
        self.tanh = nn.Tanh()

        for module_name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.001)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear_1(x)
        x = self.tanh(x)
        x = self.linear_2(x)
        mu_q, logvar_q = torch.chunk(x, chunks=2, dim=1)
        return mu_q, logvar_q


class Decoder(nn.Module):
    def __init__(self, options, p_dims=[200, 600, 20108]):
        super(Decoder, self).__init__()
        self.options = options
        self.p_dims = p_dims

        self.linear_1 = nn.Linear(self.p_dims[0], self.p_dims[1], bias=True)
        self.linear_2 = nn.Linear(self.p_dims[1], self.p_dims[2], bias=True)
        self.tanh = nn.Tanh()

        for module_name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.001)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.tanh(x)
        x = self.linear_2(x)
        return x


class MultiVAE(nn.Module):
    def __init__(
        self,
        cuda2=True,
        weight_decay=0.0,
        dropout_p=0.5,
        q_dims=[20108, 600, 200],
        p_dims=[200, 600, 20108],
        n_conditioned=0,
        sensitive_attributes=["country"],
    ):
        super(MultiVAE, self).__init__()
        self.cuda2 = cuda2
        self.weight_decay = weight_decay
        self.n_conditioned = n_conditioned
        self.q_dims = q_dims
        self.p_dims = p_dims
        self.q_dims[0] += self.n_conditioned
        self.p_dims[0] += self.n_conditioned

        self.encoder = Encoder(None, dropout_p=dropout_p, q_dims=self.q_dims)
        self.decoder = Decoder(None, p_dims=self.p_dims)

        # self.sensitive_atr_vae = SENSITIVE_ATTR_VAE(n_sensitive_attributes=n_sensitive_attributes)
        self.multi_label_classifier = MultiLabelClassifier(
            input_size=200,
            num_classes=get_summed_unique_values_count(
                user_mapper, columns=sensitive_attributes
            ),
        )

    def forward(self, x, c):
        x = f.normalize(x, p=2, dim=1)
        if self.n_conditioned > 0:
            x = torch.cat((x, c), dim=1)
        mu_q, logvar_q = self.encoder.forward(x)
        std_q = torch.exp(0.5 * logvar_q)
        KL = torch.mean(
            torch.sum(0.5 * (-logvar_q + torch.exp(logvar_q) + mu_q**2 - 1), dim=1)
        )

        if True:
            if self.training:
                epsilon = torch.randn_like(std_q, requires_grad=False)
                sampled_z = mu_q + epsilon * std_q
            else:
                epsilon = torch.randn_like(std_q, requires_grad=False)
                sampled_z = mu_q
        else:
            epsilon = torch.randn_like(std_q, requires_grad=False)
            sampled_z = mu_q + epsilon * std_q

        if self.n_conditioned > 0:
            sampled_z = torch.cat((sampled_z, c), dim=1)
        logits = self.decoder.forward(sampled_z)

        return logits, KL, mu_q, std_q, epsilon, sampled_z

    def get_l2_reg(self):
        l2_reg = Variable(torch.FloatTensor(1), requires_grad=True)
        if self.weight_decay > 0:
            for k, m in self.state_dict().items():
                if k.endswith(".weight"):
                    l2_reg = l2_reg + torch.norm(m, p=2) ** 2
        if self.cuda2:
            l2_reg = l2_reg.cuda()
        return self.weight_decay * l2_reg[0]


DS = LFM1bDataset
dt = DS("./Data/", item_mapper, user_mapper, target=["country"], split="train")
train_loader = torch.utils.data.DataLoader(
    dt,
    batch_size=500,
    shuffle=False,
)


dt = DS(
    "./Data/",
    item_mapper,
    user_mapper,
    target=["country"],
    split="valid",
)
valid_loader = torch.utils.data.DataLoader(
    dt,
    batch_size=200,
    shuffle=False,
)


class Trainer(object):
    def __init__(
        self,
        cmd,
        cuda,
        model,
        optim=None,
        train_loader=None,
        valid_loader=None,
        test_loader=None,
        log_file=None,
        interval_validate=1,
        lr_scheduler=None,
        dataset_name=None,
        gamma=0.0,
        tau=0.0,
        start_step=0,
        total_steps=1e5,
        start_epoch=0,
        bias=False,
        target=None,
        total_anneal_steps=200000,
        beta=0.1,
        do_normalize=True,
        item_mapper=None,
        user_mapper=None,
        checkpoint_dir=None,
        result_dir=None,
        print_freq=1,
        result_save_freq=1,
        checkpoint_freq=1,
        base_dir=None,
    ):
        self.cmd = cmd
        self.cuda = cuda
        self.model = model
        self.item_mapper = item_mapper
        self.user_mapper = user_mapper
        self.dataset_name = dataset_name
        self.bias = bias
        self.base_dir = base_dir

        self.optim = optim
        self.lr_scheduler = lr_scheduler

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.timestamp_start = datetime.datetime.now()

        if self.cmd == "train":
            self.interval_validate = interval_validate

        self.start_step = start_step
        self.step = start_step
        self.total_steps = total_steps
        self.epoch = start_epoch

        self.do_normalize = do_normalize
        self.print_freq = print_freq
        self.checkpoint_freq = checkpoint_freq

        self.checkpoint_dir = checkpoint_dir

        self.total_anneal_steps = total_anneal_steps
        self.beta = beta
        self.gamma = gamma
        self.tau = tau

        self.ndcg, self.recall, self.ash, self.amt, self.alt, self.ent, self.demo = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        self.loss, self.kl, self.posb, self.popb = [], [], [], []
        self.neg, self.kl, self.ubias = [], [], []

        self.target = target
        self.criterion = torch.nn.CrossEntropyLoss()
        self.classifier_criterion = torch.nn.BCELoss()

    def validate(self, cmd="valid", k=100):
        assert cmd in ["valid", "test"]
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        self.model.eval()

        end = time.time()

        n10_list, n100_list, r10_list, r100_list = [], [], [], []
        embs_list = []
        att_round, rel_round, cnt_round, pcount_round, udx_list = [], [], [], [], []
        result = []
        eval_loss = 0.0
        eval_neg = 0.0
        eval_kl = 0.0
        eval_ubias = 0.0

        loader_ = self.valid_loader if cmd == "valid" else self.test_loader

        step_counter = 0
        for batch_idx, (data_tr, data_te, prof, uindex, class_target) in tqdm.tqdm(
            enumerate(loader_),
            total=len(loader_),
            desc="{} check epoch={}, len={}".format(
                "Valid" if cmd == "valid" else "Test", self.epoch, len(loader_)
            ),
            ncols=80,
            leave=False,
        ):
            step_counter = step_counter + 1

            if self.cuda:
                data_tr = data_tr.cuda()
                prof = prof.cuda()
                class_target = class_target.cuda()
            data_tr = Variable(data_tr)
            prof = Variable(prof)
            data_time.update(time.time() - end)
            end = time.time()

            with torch.no_grad():
                logits, KL, mu_q, std_q, epsilon, sampled_z = self.model.forward(
                    data_tr, prof
                )

                log_softmax_var = f.log_softmax(logits, dim=1)
                neg_ll = -torch.mean(torch.sum(log_softmax_var * data_tr, dim=1))
                eval_neg += neg_ll.item()
                eval_kl += KL.item()

                user_bias = utils.calc_user_bias(
                    torch.sum(log_softmax_var * data_tr, dim=1), class_target
                )
                eval_ubias += user_bias.item()

                ## SENSITIVE VAE ACCURACY
                # y_hat, mean, log_var = self.model.sensitive_atr_vae(sampled_z)
                y_hat = self.model.multi_label_classifier(sampled_z)

                if self.cuda:
                    # class_loss = loss_function(sensitive_attr=Variable(sens.type(torch.FloatTensor)).cuda(), x_hat=y_hat, mean=mean, log_var=log_var)
                    class_loss = self.classifier_criterion(
                        y_hat, Variable(class_target.type(torch.FloatTensor)).cuda()
                    )
                else:
                    # class_loss = loss_function(sensitive_attr=Variable(sens.type(torch.FloatTensor)), x_hat=y_hat, mean=mean, log_var=log_var)
                    class_loss = self.classifier_criterion(
                        y_hat, Variable(class_target.type(torch.FloatTensor))
                    )

                eval_loss += class_loss.item()

                pred_val = logits.cpu().detach().numpy()
                pred_val[data_tr.cpu().detach().numpy().nonzero()] = -np.inf

                data_te_csr = sparse.csr_matrix(data_te.numpy())
                n10_list.append(
                    utils.NDCG_binary_at_k_batch(pred_val, data_te_csr, k=10)
                )
                n100_list.append(
                    utils.NDCG_binary_at_k_batch(pred_val, data_te_csr, k=100)
                )
                r10_list.append(utils.Recall_at_k_batch(pred_val, data_te_csr, k=10))
                r100_list.append(utils.Recall_at_k_batch(pred_val, data_te_csr, k=100))

                if cmd == "test":
                    for user in np.arange(data_te.numpy().shape[0]):
                        dict_out = {}
                        preds = pred_val[user, :]

                        dict_out["num_missing_terms"] = len(
                            np.array(data_te.numpy()[user, :]).nonzero()[0]
                        )
                        dict_out["missing_terms"] = " ".join(
                            [
                                str(x)
                                for x in list(
                                    np.array(data_te.numpy()[user, :]).nonzero()[0]
                                )
                            ]
                        )
                        dict_out["num_terms"] = len(
                            np.array(data_te.numpy()[user, :]).nonzero()[0]
                        ) + len(
                            np.array(data_tr.cpu().detach().numpy()[user, :]).nonzero()[
                                0
                            ]
                        )
                        dict_out["recommended_terms"] = " ".join(
                            [str(x) for x in list(np.argsort(-preds)[:k])]
                        )
                        dict_out["user_id"] = int(uindex[user].cpu().detach().numpy())
                        dict_out["scores"] = " ".join(
                            [
                                str(x)
                                for x in list(np.sort(self.softmax(preds))[::-1][:k])
                            ]
                        )
                        result.append(dict_out)

        avg_loss = eval_loss / len(loader_)
        avg_neg = eval_neg / len(loader_)
        avg_kl = eval_kl / len(loader_)
        avg_ubias = eval_ubias / len(loader_)

        metrics = []
        if cmd == "valid":
            n10_list = np.concatenate(n10_list, axis=0)
            n100_list = np.concatenate(n100_list, axis=0)
            r10_list = np.concatenate(r10_list, axis=0)
            r100_list = np.concatenate(r100_list, axis=0)

            self.ndcg.append(np.mean(n100_list))
            self.recall.append(np.mean(r100_list))
            self.loss.append(avg_loss)
            self.neg.append(avg_neg)
            self.kl.append(avg_kl)
            self.ubias.append(avg_ubias)

            np.save(
                "results/"
                + self.dataset_name
                + "_ndcg_{}_{}_{}_{}.npy".format(
                    self.target, self.beta, self.gamma, self.tau
                ),
                self.ndcg,
            )
            np.save(
                "results/"
                + self.dataset_name
                + "_recall_{}_{}_{}_{}.npy".format(
                    self.target, self.beta, self.gamma, self.tau
                ),
                self.recall,
            )
            np.save(
                "results/"
                + self.dataset_name
                + "_loss_{}_{}_{}_{}.npy".format(
                    self.target, self.beta, self.gamma, self.tau
                ),
                self.loss,
            )
            np.save(
                "results/"
                + self.dataset_name
                + "_neg_{}_{}_{}_{}.npy".format(
                    self.target, self.beta, self.gamma, self.tau
                ),
                self.neg,
            )
            np.save(
                "results/"
                + self.dataset_name
                + "_kl_{}_{}_{}_{}.npy".format(
                    self.target, self.beta, self.gamma, self.tau
                ),
                self.kl,
            )
            np.save(
                "results/"
                + self.dataset_name
                + "_ubias_{}_{}_{}_{}.npy".format(
                    self.target, self.beta, self.gamma, self.tau
                ),
                self.ubias,
            )

            # SAVE MODEL
            torch.save(
                {
                    "epoch": self.epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optim.state_dict(),
                },
                self.checkpoint_dir
                + self.dataset_name
                + "_vae_{}_{}_{}_{}.pth".format(
                    self.target, self.beta, self.gamma, self.tau
                ),
            )
            # with open(self.checkpoint_dir+self.dataset_name+'_vae_'+str(self.bias)+'_'+str(self.alpha)+'.pt', 'wb') as model_file: torch.save(self.model, model_file)
            # torch.save({'state_dict': self.model.state_dict()}, self.checkpoint_dir+'vae')

            metrics.append(
                "NDCG@10,{:.5f},{:.5f}".format(
                    np.mean(n10_list), np.std(n10_list) / np.sqrt(len(n10_list))
                )
            )
            metrics.append(
                "NDCG@100,{:.5f},{:.5f}".format(
                    np.mean(n100_list), np.std(n100_list) / np.sqrt(len(n100_list))
                )
            )
            metrics.append(
                "Recall@10,{:.5f},{:.5f}".format(
                    np.mean(r10_list), np.std(r10_list) / np.sqrt(len(r10_list))
                )
            )
            metrics.append(
                "Recall@100,{:.5f},{:.5f}".format(
                    np.mean(r100_list), np.std(r100_list) / np.sqrt(len(r100_list))
                )
            )
            print("\n" + ",\n".join(metrics))

        else:
            final_results = pd.DataFrame(result)
            final_results = final_results.merge(
                self.user_mapper[["user_id", "sex", "country", "age"]],
                on="user_id",
                how="inner",
            )
            final_results.to_csv(
                "results/{}_final_results_{}_{}_{}_{}.csv".format(
                    self.dataset_name, self.target, self.beta, self.gamma, self.tau
                ),
                index=False,
            )

        self.model.train()

    def train_epoch(self):
        cmd = "train"
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        self.model.train()

        end = time.time()
        for batch_idx, (data_tr, data_te, prof, uidx, class_target) in tqdm.tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc="Train check epoch={}, len={}".format(
                self.epoch, len(self.train_loader)
            ),
            ncols=80,
            leave=False,
        ):
            self.step += 1

            if self.cuda:
                data_tr = data_tr.cuda()
                prof = prof.cuda()
                # added by me
                class_target = class_target.cuda()
            data_tr = Variable(data_tr)
            prof = Variable(prof)
            data_time.update(time.time() - end)
            end = time.time()

            logits, KL, mu_q, std_q, epsilon, sampled_z = self.model.forward(
                data_tr, prof
            )

            log_softmax_var = f.log_softmax(logits, dim=1)
            neg_ll = -torch.mean(torch.sum(log_softmax_var * data_tr, dim=1))

            l2_reg = self.model.get_l2_reg()

            if self.total_anneal_steps > 0:
                self.anneal = min(self.beta, 1.0 * self.step / self.total_anneal_steps)
            else:
                self.anneal = self.beta

            ## SENSITIVE VAE ACCURACY
            # y_hat, mean, log_var = self.model.sensitive_atr_vae(sampled_z)
            y_hat = self.model.multi_label_classifier(sampled_z)
            if self.cuda:
                # class_loss = loss_function(sensitive_attr=Variable(sens.type(torch.FloatTensor)).cuda(), x_hat=y_hat, mean=mean, log_var=log_var)
                class_loss = self.classifier_criterion(
                    y_hat, Variable(class_target.type(torch.FloatTensor)).cuda()
                )
            else:
                # class_loss = loss_function(sensitive_attr=Variable(sens.type(torch.FloatTensor)), x_hat=y_hat, mean=mean, log_var=log_var)
                class_loss = self.classifier_criterion(
                    y_hat, Variable(class_target.type(torch.FloatTensor))
                )

            # USER BIAS
            user_bias = utils.calc_user_bias(
                torch.sum(log_softmax_var * data_tr, dim=1), class_target
            )

            loss = (
                neg_ll
                + self.anneal * KL
                + l2_reg
                - self.gamma * class_loss
                + self.tau * user_bias
            )
            print("Total loss: {}\n".format(loss / len(data_tr)))
            # backprop
            self.model.zero_grad()
            loss.backward()
            self.optim.step()

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def train(self):
        max_epoch = 100
        for epoch in tqdm.trange(0, max_epoch, desc="Train", ncols=80):
            self.epoch = epoch
            self.train_epoch()
            self.lr_scheduler.step()
            self.validate(cmd="valid")
            # self.validate(cmd='test')

    def test(self):
        self.validate(cmd="test")


cuda = torch.cuda.is_available()
if cuda:
    print("torch.backends.cudnn.version: {}".format(torch.backends.cudnn.version()))

if not os.path.isdir("./checkpoint"):
    os.mkdir("./checkpoint")
if not os.path.isdir("./results"):
    os.mkdir("./results")


cfg = dict(
    max_iteration=1000000,
    lr=1e-4,
    momentum=0.9,
    weight_decay=0.0,
    gamma=0.1,  # "lr_policy: step"
    step_size=200000,  # "lr_policy: step" e-6
    interval_validate=1000,
)


model = MultiVAE(
    dropout_p=0.5,
    weight_decay=0.0,
    cuda2=cuda,
    q_dims=[item_mapper.shape[0], 2000, 200],
    p_dims=[200, 2000, item_mapper.shape[0]],
    n_conditioned=0,
    sensitive_attributes=["country"],  # only country for now
)
# 3. optimizer
optim = torch.optim.Adam(
    [
        {
            "params": list(utils.get_parameters(model, bias=False)),
            "weight_decay": 0.0,
        },
        {
            "params": list(utils.get_parameters(model, bias=True)),
            "weight_decay": 0.0,
        },
    ],
    lr=cfg["lr"],
)
if cuda:
    model = model.cuda()
print(model)


# lr_policy: step
last_epoch = -1
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optim, milestones=[50, 75], gamma=cfg["gamma"], last_epoch=last_epoch
)


trainer = Trainer(
    cmd="train",
    cuda=cuda,
    model=model,
    optim=optim,
    gamma=0.5,
    tau=0.5,
    lr_scheduler=lr_scheduler,
    train_loader=train_loader,
    valid_loader=valid_loader,
    # test_loader=test_loader,
    start_step=0,
    total_steps=int(3e5),
    interval_validate=None,
    checkpoint_dir="./checkpoint/",
    print_freq=1,
    checkpoint_freq=1,
    total_anneal_steps=2000,
    beta=0.5,
    item_mapper=item_mapper,
    user_mapper=user_mapper,
    dataset_name="lfm2b",
    # alpha=0.5,
    base_dir="./Data/",
    target=["country"],
)
trainer.train()
