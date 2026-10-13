# 贡献指南

感谢参与本项目。以下约定旨在保证协作顺畅与结果可复现。

## 环境要求

- Python 3.10+（开发于 3.12）
- 依赖见 `requirements.txt`（锁定版本见 `constraints.txt`）

```bash
python -m pip install -r requirements.txt -c constraints.txt
# 开发/工具链（可选）
python -m pip install ruff mypy
# 数据采集/构建工具链（可选）
python -m pip install -r requirements.txt -e ".[pipeline]"
```

## 运行与测试

```bash
# 单元 + 端到端测试（标准库 unittest，无需 API key）
python -m unittest discover -s tests -v

# 静态检查（CI 同款）
ruff check .

# 类型检查（非阻塞，渐进式）
mypy file_processor
```

> 测试会在项目根写入 `.test_tmp/`（已在 `.gitignore` 中）。

## 代码约定

- 通过 `ruff`（当前规则集 `E9,F`）与保持语法正确是合并前提；建议逐步扩大规则集
  （`I` / `UP` / `B`）而非一次引入。
- 保持既有中文注释与 docstring 风格；新增逻辑请补充对应单元测试。
- 不要提交任何凭据。项目根 `.env` 已被忽略；模板见 `.env.example`。

## 提交与分支

- 提交信息使用中文，采用类型前缀：`feat:` / `fix:` / `docs:` / `chore:` / `build:` / `refactor:` / `test:`。
- 一个提交聚焦一件事；避免混入无关格式化。

## 重要：数据与目录约定

- **语料数据不入库**。`data/` 采用 `.gitignore` 白名单：
  - `data/*` 全忽略；仅 `data/README.md`、`data/_pipeline/` 下少量重建脚本与清单被反选保留。
  - 若新增需要入库的重建脚本，请在 `.gitignore` 追加 `!data/_pipeline/<文件名>`。
- 产出目录 `datasets/` 亦不入库（可由 `data/_pipeline` 重建）。

## 提交 Pull Request

1. 从 `main` / `master` 拉出特性分支。
2. 确保 `ruff check .` 与测试通过。
3. PR 描述中说明：改了什么、为什么、如何验证；涉及行为变更请附前后对比。

## 安全问题

如发现凭据泄露或安全缺陷，请勿公开提交 Issue，先行私下联系维护者。
