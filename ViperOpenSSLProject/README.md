# ViperOpenSSLProject

基于 [ViperTemplateProject](../ViperTemplateProject) 模板创建的 **OpenSSL 3.x FFI 测试工程**，
用来验证 `includes/openssl` FFI 层（摘要/哈希、EVP、证书 PKI、SSL/TLS）的链接与调用。

## 说明

- FFI 层源码在 `../includes/openssl/`（`import openssl` 使用）。
- 当前 App/main.vp 为「版本 + SHA256/MD5 摘要」演示。
- 链接目标：**已选 msys2 mingw64 预装 OpenSSL 3.6.2 动态库**（`-L D:/msys64/mingw64/lib -lssl -lcrypto`，见 project.vpj）。
  运行 `app.exe` 前需把 `D:/msys64/mingw64/bin` 加入 PATH（依赖 `libcrypto-3-x64.dll` / `libssl-3-x64.dll`）。

## 目录结构

```
ViperOpenSSLProject/
├── .gitignore
├── LICENSE
├── README.md
├── project.vpj          # 编译/链接配置（msys2 3.6.2 动态库）
└── App/
    └── main.vp          # 版本 + 摘要演示
```

## 备选：openssl_358_C 源码自编译静态库

若改用 `../includes/openssl/openssl_358_C` 自编译，先把生成的
`libcrypto.a` / `libssl.a` 填入 `linker.static_libs`（并去掉上面的 `-lcrypto -lssl`）。

## 编译运行

```powershell
cd D:\Users\TermiNexus\Desktop\TransPyC\ViperOpenSSLProject
python ..\Projectrans.py --project .\project.vpj --clean-cache=false
.\output\app.exe   # 需 D:\msys64\mingw64\bin 在 PATH（依赖 libcrypto-3-x64.dll/libssl-3-x64.dll）
```
