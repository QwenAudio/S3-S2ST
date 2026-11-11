# # -*- coding: utf-8 -*-
# import oss2
# from oss2.credentials import EnvironmentVariableCredentialsProvider
# from itertools import islice
# import os
# import logging
# import time
# import random
#
# # 配置日志
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#
# # 检查环境变量是否已设置
# required_env_vars = ['OSS_ACCESS_KEY_ID', 'OSS_ACCESS_KEY_SECRET']
# for var in required_env_vars:
# 	if var not in os.environ:
# 		logging.error(f"Environment variable {var} is not set.")
# 		exit(1)
#
# # 从环境变量中获取访问凭证
# auth = oss2.ProviderAuthV4(EnvironmentVariableCredentialsProvider())
#
# # 设置Endpoint和Region
# endpoint = "https://oss-cn-hangzhou.aliyuncs.com"
# region = "cn-hangzhou"
#
#
# def generate_unique_bucket_name():
# 	# 获取当前时间戳
# 	timestamp = int(time.time())
# 	# 生成0到9999之间的随机数
# 	random_number = random.randint(0, 9999)
# 	# 构建唯一的Bucket名称
# 	bucket_name = f"demo-{timestamp}-{random_number}"
# 	return bucket_name
#
#
# # # 生成唯一的Bucket名称
# # bucket_name = generate_unique_bucket_name()
# # bucket = oss2.Bucket(auth, endpoint, bucket_name, region=region)
#
#
# def create_bucket(bucket):
# 	try:
# 		bucket.create_bucket(oss2.models.BUCKET_ACL_PRIVATE)
# 		logging.info("Bucket created successfully")
# 	except oss2.exceptions.OssError as e:
# 		logging.error(f"Failed to create bucket: {e}")
#
#
# def upload_file(bucket, object_name, data):
# 	try:
# 		result = bucket.put_object(object_name, data)
# 		logging.info(f"File uploaded successfully, status code: {result.status}")
# 	except oss2.exceptions.OssError as e:
# 		logging.error(f"Failed to upload file: {e}")
#
#
# def download_file(bucket, object_name):
# 	try:
# 		file_obj = bucket.get_object(object_name)
# 		content = file_obj.read().decode('utf-8')
# 		logging.info("File content:")
# 		logging.info(content)
# 		return content
# 	except oss2.exceptions.OssError as e:
# 		logging.error(f"Failed to download file: {e}")
#
#
# def list_objects(bucket):
# 	try:
# 		objects = list(islice(oss2.ObjectIterator(bucket), 10))
# 		for obj in objects:
# 			logging.info(obj.key)
# 	except oss2.exceptions.OssError as e:
# 		logging.error(f"Failed to list objects: {e}")
#
#
# def delete_objects(bucket):
# 	try:
# 		objects = list(islice(oss2.ObjectIterator(bucket), 100))
# 		if objects:
# 			for obj in objects:
# 				bucket.delete_object(obj.key)
# 				logging.info(f"Deleted object: {obj.key}")
# 		else:
# 			logging.info("No objects to delete")
# 	except oss2.exceptions.OssError as e:
# 		logging.error(f"Failed to delete objects: {e}")
#
#
# def delete_bucket(bucket):
# 	try:
# 		bucket.delete_bucket()
# 		logging.info("Bucket deleted successfully")
# 	except oss2.exceptions.OssError as e:
# 		logging.error(f"Failed to delete bucket: {e}")
#
#
# # 主流程
# if __name__ == '__main__':
# 	# 1. 创建Bucket
# 	create_bucket(bucket)
# 	# 2. 上传文件
# 	upload_file(bucket, 'test-string-file', b'Hello OSS, this is a test string.')
# 	# 3. 下载文件
# 	download_file(bucket, 'test-string-file')
# 	# 4. 列出Bucket中的对象
# 	list_objects(bucket)
# 	# 5. 删除Bucket中的对象
# 	delete_objects(bucket)
# 	# 6. 删除Bucket
# 	delete_bucket(bucket)


import os
import oss2


def safe_path(file_path: str):
    return (file_path
            .replace(" ", "%20")
            .replace("(", "%28")
            .replace(")", "%29")
            .replace(":", "%3A")
            .replace("?", "%3F")
            .replace("&", "%26")
            .replace("\n", "%0A")
            .replace("/", "__")
    )


def download_file(bucket: oss2.Bucket, object_name: str, file_path: str):
    """
    下载文件到本地
    :param bucket: oss2.Bucket 实例
    :param object_name: OSS上的对象名称
    :param file_path: 本地保存路径
    :return: 下载是否成功
    """
    try:
        bucket.get_object_to_file(object_name, file_path)
        return True
    except oss2.exceptions.NoSuchKey:
        print(f"文件不存在: {object_name}")
        return None # so hacky
    except Exception as e:
        print(f"下载失败: {e}")
        return False


def download_audio_to_local(source_wav: str, file_path: str):
    from funasr.utils.load_utils import buckets, auth, endpoint, region

    _bucket_name, object_name = source_wav[len("oss://"):].split("/", maxsplit=1)
    if _bucket_name not in buckets:
        buckets[_bucket_name] = oss2.Bucket(auth, endpoint, _bucket_name, region=region)
    if not buckets[_bucket_name].object_exists(object_name):
        raise RuntimeError(f"{object_name} not found")

    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    download_file(buckets[_bucket_name], object_name, file_path)