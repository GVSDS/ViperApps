# Viper Template Project

[English](#english) | [中文](#中文)

A minimal, ready-to-use template for bootstrapping a new [Viper](../wiki/01-overview.md) language project on **Windows (x86_64-pc-windows-gnu)**.

Windows 平台（x86_64-pc-windows-gnu）下 [Viper](../wiki/01-overview.md) 语言新项目的最小可用模板。

---

<a id="中文"></a>

## 中文

### 这是什么

`ViperTemplateProject` 是基于 `Viper` 语言的 **Windows 桌面应用模板**。克隆本仓库后，你可以立即得到一个能够：

- 设置 Windows 控制台为 UTF-8 编码（避免中文乱码）
- 打印一行 `你好，世界`
- 通过 `llc + clang++`（mingw 工具链）编译为原生 `app.exe`

的最小 Viper 项目。

模板对应的语言编译器是 [TransPyC](https://git.gvsds.com/GVSDS/TransPyC)；关于 Viper 语言本身的概览，参见 [TransPyC 仓库 wiki](../wiki/01-overview.md)。

### 目录结构

```
ViperTemplateProject/
├── .gitignore           # 忽略 build/、*.o、*.exe、缓存等
├── README.md            # 本文件
├── project.vpj          # 项目配置（编译器、链接器、目标三元组等）
└── App/
    └── main.py          # 入口文件
```

### 前提条件

| 工具 | 说明 |
|------|------|
| `python` ≥ 3.10 | 用于运行 TransPyC 编译器 |
| `llc` | LLVM 静态编译器，编译 `.ll` → `.o` |
| `clang++` | mingw 工具链下的链接驱动（`x86_64-w64-mingw32`） |
| `TransPyC` 仓库 | 本编译器，克隆在 `ViperTemplateProject/..` 同一目录下 |

环境变量建议（PowerShell）：

```powershell
$env:PATH = "D:\msys64\mingw64\bin;D:\LLVM\bin;$env:PATH"
```

### 快速开始

```powershell
# 1. 确保目录布局正确
#    D:\Users\TermiNexus\Desktop\TransPyC\
#    ├── TransPyC\        # 编译器仓库
#    └── ViperTemplateProject\   # 本仓库

cd D:\Users\TermiNexus\Desktop\TransPyC\ViperTemplateProject

# 2. 编译并运行
python ..\TransPyC\Projectrans.py `
    --project .\project.vpj `
    --clean-cache=false

# 3. 产物位于 output\app.exe
.\output\app.exe
```

预期输出：

```
你好，世界
```

### 编写你的第一个程序

打开 `App/main.py`，替换 `print("你好，世界")` 为任意合法 Viper 表达式。例如：

```python
import w32.win32console as w32cmd
import t, c

pagecode: t.CDefine = 65001

@t.CExport
def main() -> int:
    w32cmd.SetConsoleOutputCP(pagecode)
    w32cmd.SetConsoleCP(pagecode)

    # 你的代码从这里开始
    a: t.CInt = 1
    b: t.CInt = 2
    print("a + b =", a + b)

    return 0
```

### 修改项目配置

打开 `project.vpj`，按需调整：

| 字段 | 含义 | 模板默认值 |
|------|------|-----------|
| `name` | 项目名（仅显示用） | `Standard Template Project` |
| `source_dir` | 源文件根目录 | `./App` |
| `temp_dir` | 中间产物目录（`.pyi` / `.stub.ll`） | `./temp` |
| `output_dir` | 最终产物目录 | `./output` |
| `compiler.cmd` | 汇编器 | `llc` |
| `compiler.flags` | 传给 `llc` 的参数 | `["-filetype=obj", "-relocation-model=pic"]` |
| `linker.cmd` | 链接器 | `clang++` |
| `linker.flags` | 链接器参数 | mingw 标准库 + `kernel32` |
| `linker.output` | 产物名 | `app.exe` |
| `includes` | 标准库搜索路径 | `["../includes"]` |
| `target.triple` | LLVM 目标三元组 | `x86_64-pc-windows-gnu` |
| `target.datalayout` | LLVM 数据布局 | mingw 默认 |
| `options.target` | 后端目标 | `llvm` |
| `options.strict_mode` | 严格类型检查 | `true` |
| `options.slice_level` | 切片级别 | `3` |

### 跨平台使用

本模板默认针对 **Windows + mingw**。若需在其他平台使用，修改 `project.vpj` 的 `compiler.cmd` / `linker.cmd` / `target.triple`：

- **Linux (x86_64)**：`triple = "x86_64-unknown-linux-gnu"`，`linker.cmd = "clang++"`
- **macOS (x86_64)**：`triple = "x86_64-apple-darwin*"`，`linker.cmd = "clang++"`
- **裸机 x86_64 (ViperOS)**：`triple = "x86_64-none-elf"`，`linker.cmd = "ld.lld"`，需要 `isr.s` 启动文件

### 常见问题

**Q: 报错 `cannot find includes/os` / `cannot find includes/atom` 之类？**
A: 检查 `project.vpj` 中的 `includes` 字段是否指向 `TransPyC/includes`。`../includes` 假设本仓库与 `TransPyC` 仓库同级。

**Q: 输出 `app.exe` 双击闪退？**
A: 在终端中运行 `.\output\app.exe` 而非双击，并确认 mingw 与 LLVM 在 `PATH` 中。

**Q: 想完全清理构建产物？**
A: 使用 `python ..\TransPyC\Projectrans.py --project .\project.vpj --clean-cache` 清理缓存；删除 `output/` 和 `temp/` 即可彻底重置。

**Q: `StandardTemplateProject` 和本仓库是什么关系？**
A: 本仓库 `ViperTemplateProject` 是 `StandardTemplateProject` 的独立仓库版本。如果你仍想保留 `StandardTemplateProject` 放在 `TransPyC` 仓库内，只需把它加入 `TransPyC/.gitignore` 的 `# 排除独立仓库的目录` 小节。

### 进一步阅读

- [Viper 语言概览](../wiki/01-overview.md)
- [类型系统](../wiki/02-type-system.md)
- [OOP 与对象模型](../wiki/06-oop.md)
- [自举路线图](../wiki/13-bootstrapping.md)

---

<a id="english"></a>

## English

### What is this

`ViperTemplateProject` is a **Windows desktop application template** based on the [Viper](../wiki/01-overview.md) language. After cloning, you immediately have a minimal Viper project that:

- Sets the Windows console to UTF-8 (avoids mojibake for non-ASCII output)
- Prints `你好，世界`
- Compiles to a native `app.exe` via `llc + clang++` (mingw toolchain)

The corresponding compiler is [TransPyC](https://git.gvsds.com/GVSDS/TransPyC). For a Viper language overview, see the [TransPyC wiki](../wiki/01-overview.md).

### Directory layout

```
ViperTemplateProject/
├── .gitignore           # ignores build/, *.o, *.exe, caches, etc.
├── README.md            # this file
├── project.vpj          # project config (compiler, linker, target triple, ...)
└── App/
    └── main.py          # entry point
```

### Prerequisites

| Tool | Notes |
|------|-------|
| `python` ≥ 3.10 | runs the TransPyC compiler |
| `llc` | LLVM static compiler, `.ll` → `.o` |
| `clang++` | mingw toolchain driver (`x86_64-w64-mingw32`) |
| `TransPyC` repo | the compiler, cloned as a sibling of `ViperTemplateProject` |

Suggested `PATH` (PowerShell):

```powershell
$env:PATH = "D:\msys64\mingw64\bin;D:\LLVM\bin;$env:PATH"
```

### Quick start

```powershell
# 1. Make sure the layout is right
#    D:\Users\TermiNexus\Desktop\TransPyC\
#    ├── TransPyC\             # compiler repo
#    └── ViperTemplateProject\ # this repo

cd D:\Users\TermiNexus\Desktop\TransPyC\ViperTemplateProject

# 2. Build and run
python ..\TransPyC\Projectrans.py `
    --project .\project.vpj `
    --clean-cache=false

# 3. Output is at output\app.exe
.\output\app.exe
```

Expected output:

```
你好，世界
```

### Writing your first program

Open `App/main.py` and replace `print("你好，世界")` with any valid Viper expression. Example:

```python
import w32.win32console as w32cmd
import t, c

pagecode: t.CDefine = 65001

@t.CExport
def main() -> int:
    w32cmd.SetConsoleOutputCP(pagecode)
    w32cmd.SetConsoleCP(pagecode)

    # your code goes here
    a: t.CInt = 1
    b: t.CInt = 2
    print("a + b =", a + b)

    return 0
```

### Editing project.vpj

| Field | Meaning | Template default |
|-------|---------|------------------|
| `name` | display name | `Standard Template Project` |
| `source_dir` | source root | `./App` |
| `temp_dir` | intermediate artifacts (`.pyi` / `.stub.ll`) | `./temp` |
| `output_dir` | final artifacts | `./output` |
| `compiler.cmd` | assembler | `llc` |
| `compiler.flags` | flags passed to `llc` | `["-filetype=obj", "-relocation-model=pic"]` |
| `linker.cmd` | linker | `clang++` |
| `linker.flags` | linker flags | mingw stdlib + `kernel32` |
| `linker.output` | output name | `app.exe` |
| `includes` | standard library search paths | `["../includes"]` |
| `target.triple` | LLVM target triple | `x86_64-pc-windows-gnu` |
| `target.datalayout` | LLVM data layout | mingw default |
| `options.target` | backend target | `llvm` |
| `options.strict_mode` | strict type checking | `true` |
| `options.slice_level` | slice level | `3` |

### Cross-platform use

This template targets **Windows + mingw** by default. To use on another platform, adjust `compiler.cmd` / `linker.cmd` / `target.triple` in `project.vpj`:

- **Linux (x86_64)**: `triple = "x86_64-unknown-linux-gnu"`, `linker.cmd = "clang++"`
- **macOS (x86_64)**: `triple = "x86_64-apple-darwin*"`, `linker.cmd = "clang++"`
- **Bare-metal x86_64 (ViperOS)**: `triple = "x86_64-none-elf"`, `linker.cmd = "ld.lld"`, requires `isr.s` startup

### FAQ

**Q: Error like `cannot find includes/os` or `cannot find includes/atom`?**
A: Check that the `includes` field in `project.vpj` points to the `TransPyC/includes` directory. `../includes` assumes this repo lives next to `TransPyC`.

**Q: `app.exe` flashes and closes on double-click?**
A: Run `.\output\app.exe` from a terminal, not by double-clicking. Make sure mingw and LLVM are on `PATH`.

**Q: How to fully clean build artifacts?**
A: `python ..\TransPyC\Projectrans.py --project .\project.vpj --clean-cache` clears the cache. To reset everything, also delete `output/` and `temp/`.

**Q: What is the relationship with `StandardTemplateProject`?**
A: This repo `ViperTemplateProject` is the standalone-repo version of `StandardTemplateProject`. If you still want `StandardTemplateProject` to live inside the `TransPyC` repo, add it to the `# 排除独立仓库的目录` section in `TransPyC/.gitignore`.

### Further reading

- [Viper language overview](../wiki/01-overview.md)
- [Type system](../wiki/02-type-system.md)
- [OOP and object model](../wiki/06-oop.md)
- [Bootstrapping roadmap](../wiki/13-bootstrapping.md)
