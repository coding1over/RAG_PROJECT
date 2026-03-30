# 项目更新记录（2026-03-31）
## 核心改动
1. 修复 Milvus 连接报错：langchain-milvus 与 pymilvus 2.6+ 版本不兼容，彻底弃用
2. 重构 Milvus 操作：使用官方原生 MilvusClient，稳定无连接BUG
3. 修复 BM25 索引：稀疏向量字段 metric_type 修正为 BM25，适配 Milvus 2.5.6
4. 优化数据插入逻辑：自动生成稠密向量，BM25稀疏向量自动计算
5. 解决所有报错：KeyError/ConnectionNotExistException/索引类型错误

## 功能状态
✅ Milvus 集合创建正常
✅ 文档向量插入正常
✅ BM25 + 稠密向量混合检索正常
✅ 过滤查询功能正常