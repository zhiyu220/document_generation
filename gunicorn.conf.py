import multiprocessing

# Worker 设置
workers = 2  # 减少 worker 数量以降低内存使用
worker_class = 'sync'
worker_connections = 1000
timeout = 300  # 增加超时时间到 300 秒

# 内存相关
max_requests = 1000
max_requests_jitter = 50

# 日志设置
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# 限制请求大小
limit_request_line = 4096
limit_request_fields = 100
limit_request_field_size = 8190

# 性能优化
keepalive = 5
threads = 2
worker_tmp_dir = '/dev/shm'  # 使用内存文件系统来减少 I/O

# 优雅重启
graceful_timeout = 120 