# Dockerfile.builder (V15.0 - "CentOS 7 终极版")
# 功能：在一个CentOS 7的基础上，以最可靠的方式，安装GCC 9, Python 3.10, 和最新的CMake。

# 步骤 1: 使用与您生产环境一致的CentOS 7作为基础
FROM centos:7

# 步骤 2: 配置YUM源，确保稳定可靠
RUN rpm --import http://vault.centos.org/7.9.2009/os/x86_64/RPM-GPG-KEY-CentOS-7 && \
    rpm --import https://dl.fedoraproject.org/pub/epel/RPM-GPG-KEY-EPEL-7 && \
    rpm --import https://www.centos.org/keys/RPM-GPG-KEY-CentOS-SIG-SCLo
RUN rm -f /etc/yum.repos.d/*.repo
RUN echo -e "[base]\nname=CentOS-7 - Base\nbaseurl=http://vault.centos.org/7.9.2009/os/\$basearch/\ngpgcheck=1\ngpgkey=file:///etc/pki/rpm-gpg/RPM-GPG-KEY-CentOS-7" > /etc/yum.repos.d/CentOS-Vault.repo
RUN echo -e "[updates]\nname=CentOS-7 - Updates\nbaseurl=http://vault.centos.org/7.9.2009/updates/\$basearch/\ngpgcheck=1\ngpgkey=file:///etc/pki/rpm-gpg/RPM-GPG-KEY-CentOS-7" >> /etc/yum.repos.d/CentOS-Vault.repo
RUN echo -e "[extras]\nname=CentOS-7 - Extras\nbaseurl=http://vault.centos.org/7.9.2009/extras/\$basearch/\ngpgcheck=1\ngpgkey=file:///etc/pki/rpm-gpg/RPM-GPG-KEY-EPEL-7" >> /etc/yum.repos.d/epel.repo
RUN echo -e "[centos-sclo-rh]\nname=CentOS-7 - SCLo rh\nbaseurl=http://vault.centos.org/7.9.2009/sclo/\$basearch/rh/\nenabled=1\ngpgcheck=1\ngpgkey=file:///etc/pki/rpm-gpg/RPM-GPG-KEY-CentOS-SIG-SCLo" > /etc/yum.repos.d/CentOS-SCLo-scl-rh.repo
RUN yum clean all && yum makecache

# 步骤 3: 安装所有必要的系统级和编译依赖
RUN yum groupinstall -y 'Development Tools' && \
    yum install -y centos-release-scl devtoolset-9-gcc* \
                   bzip2-devel libffi-devel openssl-devel wget unzip && \
    yum clean all

# 步骤 4: 手动安装并链接一个现代化的CMake
WORKDIR /usr/src
COPY wheelhouse/pkg/cmake-3.28.1-linux-x86_64.tar.gz .
RUN tar -xzvf cmake-3.28.1-linux-x86_64.tar.gz && \
    ln -s /usr/src/cmake-3.28.1-linux-x86_64/bin/cmake /usr/bin/cmake && \
    ln -s /usr/src/cmake-3.28.1-linux-x86_64/bin/ctest /usr/bin/ctest && \
    ln -s /usr/src/cmake-3.28.1-linux-x86_64/bin/cpack /usr/bin/cpack

# 步骤 5: 从源代码编译并安装Python 3.10
COPY wheelhouse/pkg/Python-3.10.14.tgz .
RUN tar xzf Python-3.10.14.tgz && \
    /bin/bash -c "source /opt/rh/devtoolset-9/enable && \
                  cd /usr/src/Python-3.10.14 && \
                  ./configure --enable-optimizations && \
                  make -j$(nproc) && \
                  make altinstall" && \
    ln -s /usr/local/bin/python3.10 /usr/bin/python3 && \
    ln -s /usr/local/bin/pip3.10 /usr/bin/pip3

# 步骤 6: 将我们手动下载的所有依赖库和onnxruntime仓库全部复制并解压到容器中
COPY wheelhouse/pkg ./pkg
WORKDIR /usr/src/pkg
RUN for f in *.zip; do unzip -o "$f"; done
RUN for f in *.tar.gz; do tar -xzvf "$f"; done

# 步骤 7: 设置默认启动命令，确保我们进入的是一个激活了所有新环境的交互式Shell
CMD ["/bin/bash", "-c", "source /opt/rh/devtoolset-9/enable; exec /bin/bash"]
