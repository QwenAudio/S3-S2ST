import logging

import torchaudio
import os

try:
	import alibabacloud_oss_v2 as oss
	
	# 从环境变量中加载凭证信息（AccessKeyId 和 AccessKeySecret）
	credentials_provider = oss.credentials.EnvironmentVariableCredentialsProvider()
	
	# 加载 SDK 的默认配置
	cfg = oss.config.load_default()
	
	# 设置凭证提供者
	cfg.credentials_provider = credentials_provider
	
	# 设置存储空间所在的地域
	cfg.region = os.environ.get("OSS_REGION", None)
	
	# 如果用户提供了自定义的 endpoint，则设置到配置中
	# if args.endpoint is not None:
	#     cfg.endpoint = args.endpoint
	cfg.endpoint = os.environ.get("OSS_ENDPOINT", None)
	
	# 使用配置对象初始化 OSS 客户端
	client = oss.Client(cfg)
	
	# bucket_name = "example-storage-6"  # 指定目标存储空间的名称
	bucket_name = os.environ.get("OSS_BUCKET_NAME", None)

except ImportError:
	logging.warning("OSS not installed, please intall oss2.")
	
except AttributeError:
	logging.warning("Environment variable of OSS not set.")




def oss_remote_torchaudio_load(file_path: str):
	# bucket_name = oss_path.split("/")[2]
	# object_name = "/".join(oss_path.split("/")[3:])
	#
	if file_path.startswith("/mnt/spider-data"):
		file_path = file_path.replace("/mnt/spider-data", "spider-data/aicheng")
		object_name = file_path
	elif file_path.startswith("/mnt/spider-resource/"):
		file_path = file_path.replace("/mnt/spider-resource/", "")
		object_name = file_path
	elif file_path.startswith("/mnt/spider-resource-nichongjia/"):
		file_path = file_path.replace("/mnt/spider-resource-nichongjia/", "")
		object_name = file_path
	elif file_path.startswith("/mnt/"):
		file_path = file_path.replace("/mnt/", "")
		object_name = file_path
	else:
		object_name = file_path[len("oss:" + bucket_name):]
	
	result = client.open_file(
		bucket=bucket_name,
		key=object_name,
		block_size=1024,
		chunk_size=1024,
		enable_prefetch=False
	)
	

	return torchaudio.load(result)


