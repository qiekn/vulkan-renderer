# Tutorial sources (local mirror)

官方教程 [docs.vulkan.org](https://docs.vulkan.org/tutorial/latest/Building_a_Simple_Engine/introduction.html) 的 Cloudflare 挡了 WebFetch 和无头 `curl`，所以这里存一份从 [KhronosGroup/Vulkan-Tutorial](https://github.com/KhronosGroup/Vulkan-Tutorial) 仓库直接拉的 AsciiDoc 源文件。目录结构对应上游 `en/Building_a_Simple_Engine/`。

## 拉取

在 `docs/` 目录下运行：

```bash
./fetch.sh Engine_Architecture                  # 整章
./fetch.sh Camera_Transformations               # 下一章
./fetch.sh Engine_Architecture/05_rendering_pipeline.adoc  # 单文件
```

## 已同步章节

| 章节 | 本地路径 |
| --- | --- |
| Series Introduction | [`introduction.adoc`](./introduction.adoc) |
| Engine Architecture · Introduction | [`Engine_Architecture/01_introduction.adoc`](./Engine_Architecture/01_introduction.adoc) |
| Engine Architecture · Architectural patterns | [`Engine_Architecture/02_architectural_patterns.adoc`](./Engine_Architecture/02_architectural_patterns.adoc) |
| Engine Architecture · Component systems | [`Engine_Architecture/03_component_systems.adoc`](./Engine_Architecture/03_component_systems.adoc) |
| Engine Architecture · Resource management | [`Engine_Architecture/04_resource_management.adoc`](./Engine_Architecture/04_resource_management.adoc) |
| Engine Architecture · Rendering pipeline | [`Engine_Architecture/05_rendering_pipeline.adoc`](./Engine_Architecture/05_rendering_pipeline.adoc) |
| Engine Architecture · Event systems | [`Engine_Architecture/06_event_systems.adoc`](./Engine_Architecture/06_event_systems.adoc) |
| Engine Architecture · Conclusion | [`Engine_Architecture/conclusion.adoc`](./Engine_Architecture/conclusion.adoc) |

后续章节用 `./fetch.sh` 追加。
