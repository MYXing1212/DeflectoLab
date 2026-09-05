#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
layout(location = 2) in vec3 aNormal;

uniform mat4 uMVP;
uniform mat3 uNormalMat;   // 法向到视图空间(取视图矩阵旋转部分)
uniform float uPointSize;

out vec3 vColor;
out vec3 vNormalView;

void main()
{
    gl_Position = uMVP * vec4(aPos, 1.0);
    gl_PointSize = uPointSize;
    vColor = aColor;
    vNormalView = uNormalMat * aNormal;
}
