#version 330 core
in vec3 vColor;
in vec3 vNormalView;
out vec4 FragColor;

uniform int  uUseDepthColor;   // 1: 顶点(深度)颜色, 0: 纯色
uniform vec3 uSolidColor;      // 纯色 (0~1)
uniform int  uLighting;        // 1: 开启光照着色

void main()
{
    vec3 base = (uUseDepthColor == 1) ? vColor : uSolidColor;
    if (uLighting == 1)
    {
        vec3 N = normalize(vNormalView);
        if (N.z < 0.0) N = -N;                 // 双面: 法向朝向观察者
        vec3 L = normalize(vec3(0.35, 0.45, 1.0));  // 头灯(略偏), 增强立体感
        float diff = max(dot(N, L), 0.0);
        float amb = 0.35;
        float shade = amb + (1.0 - amb) * diff;
        base *= shade;
    }
    FragColor = vec4(base, 1.0);
}
