import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--id', type=str, default='test')
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--step', type=int, default=0)

parser.add_argument('--vocab', default='./data/vocab.pkl')
parser.add_argument('--train', default='./data/dataset_train.json')
parser.add_argument('--resnet_feat_dir', default='/home/data_ti4_c/chengkz/data/resnet_feat')
parser.add_argument('--val_ro', default='./data/dataset_ro_val.json')
parser.add_argument('--val_fu', default='./data/dataset_fu_val.json')
parser.add_argument('--test_ro', default='./data/dataset_ro_test.json')
parser.add_argument('--test_fu', default='./data/dataset_fu_test.json')
parser.add_argument('--val_pos', default='./data/dataset_pos_val.json')
parser.add_argument('--val_neg', default='./data/dataset_neg_val.json')
parser.add_argument('--test_pos', default='./data/dataset_pos_test.json')
parser.add_argument('--test_neg', default='./data/dataset_neg_test.json')

parser.add_argument('--save_loss_freq', type=int, default=20)
parser.add_argument('--save_model_freq', type=int, default=10000)
parser.add_argument('--log_dir', default='/home/chengkz/checkpoints/MultiStyle_IC_v3/log/{}')
parser.add_argument('--recheck_model_path', default='./models/Discriminator/model_3800.pt')

parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--fixed_len', type=int, default=20)
parser.add_argument('--fixed_len_o', type=int, default=5)
parser.add_argument('--fixed_len_s', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--style_lr', type=float, default=1e-4)
parser.add_argument('--unk_rate', type=float, default=0.0)
parser.add_argument('--kl_rate', type=float, default=0.035)
parser.add_argument('--cont_rate', type=float, default=0.1)
parser.add_argument('--style_rate', type=float, default=2.0)
parser.add_argument('--grad_clip', type=float, default=0.1)
parser.add_argument('--temperature', type=float, default=0.10)
parser.add_argument('--beam_num', type=int, default=10)
parser.add_argument('--beam_alpha', type=float, default=1.0)

parser.add_argument('--align_dim', type=int, default=1024)
parser.add_argument('--embed_dim', type=int, default=1024)
parser.add_argument('--hidden_dim', type=int, default=1024)
parser.add_argument('--latent_dim', type=int, default=100)
parser.add_argument('--image_dim', type=int, default=2048)
parser.add_argument('--styles_num', type=int, default=5)

parser.add_argument('--finetune', type=bool, default=False)
parser.add_argument('--pretrain_id', type=str, default='')
parser.add_argument('--pretrain_step', type=int, default=0)
parser.add_argument('--vis_mode', type=str, default="vis")

config = parser.parse_args()

