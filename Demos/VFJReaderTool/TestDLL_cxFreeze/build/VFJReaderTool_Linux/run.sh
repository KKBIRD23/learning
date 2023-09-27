# 查看是否有中海车道进程海在偷偷跑，有就干掉！
for pid in $(pgrep -f "/cstcdisk/cstclane/CSTCUpdate"); do
    cmd=$(ps -o cmd "${pid}"|grep -v "CMD")
    if kill "${pid}";then
        echo "Process Completet."
    else
        echo "failed to stop it!"
    fi
done

./VFJReaderTool
