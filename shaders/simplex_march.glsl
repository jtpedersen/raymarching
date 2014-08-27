#version 330 core

in VS_OUT {
  float time;
} fs_in;

out vec4 color;

float thresshold = 0.0f;        /* MAKE UNIFORM */

//
// Description : Array and textureless GLSL 2D/3D/4D simplex 
//               noise functions.
//      Author : Ian McEwan, Ashima Arts.
//  Maintainer : ijm
//     Lastmod : 20110822 (ijm)
//     License : Copyright (C) 2011 Ashima Arts. All rights reserved.
//               Distributed under the MIT License. See LICENSE file.
//               https://github.com/ashima/webgl-noise
// 

vec3 mod289(vec3 x) {
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 mod289(vec4 x) {
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 permute(vec4 x) {
     return mod289(((x*34.0)+1.0)*x);
}

vec4 taylorInvSqrt(vec4 r)
{
  return 1.79284291400159 - 0.85373472095314 * r;
}

/* return distance */
float snoise(vec3 v, vec3 d) { 
  const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
  const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

// First corner
  vec3 i  = floor(v + dot(v, C.yyy) );
  vec3 x0 =   v - i + dot(i, C.xxx) ;

// Other corners
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min( g.xyz, l.zxy );
  vec3 i2 = max( g.xyz, l.zxy );

  //   x0 = x0 - 0.0 + 0.0 * C.xxx;
  //   x1 = x0 - i1  + 1.0 * C.xxx;
  //   x2 = x0 - i2  + 2.0 * C.xxx;
  //   x3 = x0 - 1.0 + 3.0 * C.xxx;
  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
  vec3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y

// Permutations
  i = mod289(i); 
  vec4 p = permute( permute( permute( 
             i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0 )) 
           + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

// Gradients: 7x7 points over a square, mapped onto an octahedron.
// The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
  float n_ = 0.142857142857; // 1.0/7.0
  vec3  ns = n_ * D.wyz - D.xzx;

  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)

  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

  vec4 x = x_ *ns.x + ns.yyyy;
  vec4 y = y_ *ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);

  vec4 b0 = vec4( x.xy, y.xy );
  vec4 b1 = vec4( x.zw, y.zw );

  //vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;
  //vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;
  vec4 s0 = floor(b0)*2.0 + 1.0;
  vec4 s1 = floor(b1)*2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

  vec3 p0 = vec3(a0.xy,h.x);
  vec3 p1 = vec3(a0.zw,h.y);
  vec3 p2 = vec3(a1.xy,h.z);
  vec3 p3 = vec3(a1.zw,h.w);

//Normalise gradients
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;

// Mix final noise value
  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  float value =  42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), 
                                dot(p2,x2), dot(p3,x3) ) );


  if (value < -1.5) return 0.0f;

  /* use gradient to estimate distance */

  /* start with brute force */
  return min(0.1, .7 * dot(m, vec4(d, 0)));
  /* return .1; */
  }



void main() {
  float x = (gl_FragCoord.x) / 400.0 - 1.0;
  float y = (gl_FragCoord.y) / 300.0 - 1.0;
  float t = fs_in.time;

  vec3 v0 = vec3(5.0 * x, y, 0); // ray intersection with view plane
  vec3 p = vec3(0, 0, -2); // camera position

  /* movde */
  vec3 movement = t * vec3(0,0,.1);
  v0 += movement;
  p += movement;
  
  vec3 vd = normalize(v0 - p); // ray direction

  vec3 lightPos = vec3(70 * cos(fs_in.time * 1.7), 70 * sin(fs_in.time * 0.35), 70 * sin(fs_in.time));

  // epsilon vectors used for calculating surface normals
  float e = 0.001;    
  vec3 dx = vec3(e, 0, 0);
  vec3 dy = vec3(0, e, 0);
  vec3 dz = vec3(0, 0, e);


  // ray marching loop
  bool hit = false;
  float maxZ = 3000.0;
  
  /* color =  10 * vec4(1,1,1,1) * snoise(v0, vd); */

  
  p += vd;
  float s = 0;
  float maxstep = 80;
  while (!hit && p.z < maxZ && s < maxstep) {
      float d = snoise(p, vd);
    if (d <= 0.00005) {
      hit = true;
    } else {
      p += vd * d;
      s += 1;

    }
  }


  if (s < 1) color = vec4(1.0, 0, 0, 1);
  else if (s < 2) color = vec4(0, 1, 0, 1);
  else if (s < 3) color = vec4(0, 0, 1, 1);
  else
  color = (1.0 - s/maxstep) * vec4(1, 1,1 ,1);
  
  color  = vec4(p,1);


  /* // shading  */
  /* if (hit) { */

  /*   /\* vec3 n = normalize(vec3( *\/ */
  /*   /\*     		    dist(p + dx, spherePos) - dist(p - dx, spherePos), *\/ */
  /*   /\*     		    dist(p + dy, spherePos) - dist(p - dy, spherePos), *\/ */
  /*   /\*     		    dist(p + dz, spherePos) - dist(p - dz, spherePos) *\/ */
  /*   /\*     		    )); *\/ */

  /*   /\* float light = doLight(vd, p, n, lightPos); *\/ */

  /*   /\* float fog = clamp(pow(1.1 * p.z / maxZ, 2), 0, 1); *\/ */
  /*   /\* vec3 fogcol = vec3(0.2, 0.2, 0.2); *\/ */

  /*   /\* color = vec4(mix(vec3(light), fogcol, fog), 1); *\/ */
  /* } else { */
  /*   color = vec4(0.2, 0.2, 0.2, 1); */
  /* } */

}
