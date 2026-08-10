// ListDisplay frontend: populates the native `filename` combo widget from
// /rebalance_pack/list_files (filtered by directory_path + ext) and adds a
// refresh button that re-queries the route.

const { app } = window.comfyAPI.app;

const LIST_DISPLAY_NAME = "ListDisplay";
const ROUTE = "/rebalance_pack/list_files";

async function fetchFileList(directoryPath, ext) {
    const qs = new URLSearchParams();
    if (directoryPath) qs.set("directory_path", directoryPath);
    if (ext) qs.set("ext", ext);
    const url = `${ROUTE}?${qs.toString()}`;
    try {
        const res = await fetch(url, { cache: "no-store" });
        if (!res.ok) return [];
        const data = await res.json();
        return Array.isArray(data) ? data : [];
    } catch (e) {
        console.warn("[ListDisplay] fetch failed:", e);
        return [];
    }
}

function getWidget(node, name) {
    return node.widgets && node.widgets.find(w => w.name === name);
}

function setComboOptions(widget, options) {
    if (!widget) return;
    widget.options = widget.options || {};
    widget.options.values = options;
    if (!options.includes(widget.value)) {
        widget.value = options.length ? options[0] : "";
    }
    if (widget.callback) widget.callback(widget.value);
}

async function refreshList(node) {
    const dirWidget = getWidget(node, "directory_path");
    const extWidget = getWidget(node, "ext");
    const fileWidget = getWidget(node, "filename");
    if (!fileWidget) return;
    const dir = dirWidget ? dirWidget.value : "";
    const ext = extWidget ? extWidget.value : "";
    const options = await fetchFileList(dir, ext);
    setComboOptions(fileWidget, options);
    node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "ComfyUI-ListDisplay",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== LIST_DISPLAY_NAME) return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;

            this.addWidget("button", "Refresh", null, () => {
                refreshList(this);
            });

            setTimeout(() => refreshList(this), 0);
            return r;
        };

        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = origOnConfigure ? origOnConfigure.apply(this, arguments) : undefined;
            setTimeout(() => refreshList(this), 0);
            return r;
        };
    },
});
