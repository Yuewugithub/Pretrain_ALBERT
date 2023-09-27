import os
import math
import torch
from torch.nn import CrossEntropyLoss
from tqdm.auto import tqdm
from transformers import AlbertForMaskedLM, AlbertTokenizer, AutoTokenizer, AdamW, get_scheduler


class Trainer(object):

    def __init__(self, config):
        self.config = config
        self.tokenizer = AlbertTokenizer.from_pretrained(self.config.initial_pretrain_tokenizer)

    def train(self, train_loader, valid_loader):
        """
        预训练模型
        """
        print('training start')
        device = torch.device(self.config.device)

        # 初始化模型和优化器
        print('model loading')
        model = AlbertForMaskedLM.from_pretrained(self.config.initial_pretrain_model)
        optimizer = AdamW(model.parameters(), lr=self.config.learning_rate)

        # 定义优化器配置
        num_training_steps = self.config.num_epochs * len(train_loader)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=self.config.num_warmup_steps,
            num_training_steps=num_training_steps
        )

        model.to(device)
        print('start to train')
        model.train()
        progress_bar = tqdm(range(num_training_steps))
        loss_best = math.inf

        for epoch in range(self.config.num_epochs):
            for i, batch in enumerate(train_loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                # 计算loss
                loss = outputs.loss
                loss = loss.mean()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

                if i % 500 == 0:
                    print('epoch:{0}  iter:{1}/{2}  loss:{3}'.format(epoch, i, len(train_loader), loss.item()))
            # 模型保存
            self.eval(valid_loader, model, epoch, device)
            model_save = model.module if torch.cuda.device_count() > 1 else model
            path = self.config.path_model_save + 'epoch_{}/'.format(epoch)
            model_save.save_pretrained(path)


    def eval(self, eval_dataloader, model, epoch, device):
        losses = []
        model.eval()

        input = []
        label = []
        pred = []
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            loss = loss.mean()
            losses.append(loss)

            # 生成包含[MASK]标记的文本
            input_ids = batch['input_ids']
            mask_indices = (input_ids == self.tokenizer.mask_token_id).nonzero()  # 找到[MASK]标记的位置
            generated_texts = []
            for i in range(len(input_ids)):
                masked_ids = input_ids[i].tolist()
                for mask_index in mask_indices[i]:
                    masked_ids[mask_index] = self.tokenizer.mask_token_id
                generated_text = self.tokenizer.decode(masked_ids, skip_special_tokens=True)
                generated_texts.append(generated_text)

            tmp_src = batch['input_ids'].cpu().numpy()
            tmp_label = batch['labels'].cpu().numpy()
            tmp_pred = outputs.logits.cpu().numpy()
            for i in range(len(tmp_label)):
                line_l = tmp_label[i]
                line_l_split = [ x for x in line_l if x not in [0]]
                line_s = tmp_src[i]
                line_s_split = line_s[:len(line_l_split)]
                line_p = tmp_pred[i]
                line_p_split = line_p[:len(line_l_split)]
                tmp_s = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(line_s_split))
                tmp_lab = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(line_l_split))
                tmp_p = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(line_p_split))
                input.append(tmp_s)
                label.append(tmp_lab)
                pred.append(tmp_p)
        # 计算困惑度
        losses = torch.stack(losses)
        losses_avg = torch.mean(losses)
        perplexity = math.exp(losses_avg)
        print('eval {0}: loss:{1}  perplexity:{2}'.format(epoch, losses_avg.item(), perplexity))
        for i in range(10):
            print('-' * 30)
            print('input: {}'.format(input[i]))
            print('label: {}'.format(label[i]))
            print('pred : {}'.format(pred[i]))

        return losses_avg
if __name__ == '__main__':
    # 请确保您的Config类以及训练和评估数据集的准备和加载等部分已正确设置
    config = Config()
    trainer = Trainer(config)
    trainer.train(train_loader, valid_loader)