import * as k from './vendor/iso-kit.mjs';
import * as A from './vendor/audit.mjs';

// One reusable 41-part console. The changing SVG is displayed on the actual
// glass plane; it is deliberately not recorded as physical solid geometry.
export function motionInstrument(id,body='',audit=false){
 const base={x:-160,y:-100,w:320,d:200,r:8};
 const P=k.fitProjection(k.boxCorners(base,-15,19),640,400,{pad:26,azimuth:80,elevation:58});
 const R=A.recorder(P);let seq=0;
 const at=p=>k.iso(p,P);
 function part(name,x,y,w,d,z,h,extra={}){
  const plan={x,y,w,d,r:Math.min(1.8,w/4,d/4)},shape=A.slab(plan,z,h,{name,ground:!!extra.ground},6);
  const detail=extra.detail??'';
  const svg=k.solidSvg(k.slabOf(plan,z,h,P,6,Math.min(.6,h/4)),{tone:extra.tone??'mid',inner:detail,lit:name==='cursor'});
  R.put({name,at:[x+w/2,y+d/2,z+h/2],piece:name==='cursor'?'cursor':'',move:name==='cursor'?[240,0,0]:[0,0,0],shapes:[R.solid(shape)],svg:`<g ${name==='cursor'?'data-motion-cursor':''}>${svg}</g>`});
 }
 for(const[x,y]of k.corners(base,15))part('stand.foot'+seq++,x-6,y-6,12,12,-15,5,{ground:true,tone:'lo'});
 part('base',-160,-100,320,200,-10,10,{tone:'lo',detail:k.lineSvg(k.planOutline(k.insetPlan(base,4),0,P),{tone:'lo'})});
 part('cover',-157,-97,314,194,0,5,{tone:'lo',detail:k.dotsSvg(k.corners(base,9).map(([x,y])=>at([x,y,5])),{size:.55,tone:'hi'})});
 const screen={x:-139,y:-74,w:278,d:150.5833333333};
 // Glass is supported on the cover, with a separate bezel around each edge.
 part('glass',screen.x,screen.y,screen.w,screen.d,5,1,{tone:'hi'});
 part('bezel.back',-143,-78,286,4,5,3);
 part('bezel.front',-143,screen.y+screen.d,286,4,5,3);
 part('bezel.left',-143,-74,4,screen.d,5,3);
 part('bezel.right',139,-74,4,screen.d,5,3);
 for(let n=0;n<8;n++)part('fin'+n,-70+n*19,-92,12,9,5,3,{tone:'lo'});
 for(const x of [-153,146]){
  part('connector.carrier'+x,x,-63,7,98,5,1,{tone:'lo'});
  for(let n=0;n<6;n++)part('contact'+seq++,x+1,-58+n*15,5,7,6,1,{tone:'mid'});
 }
 for(const x of [-133,-111,98,120])part('control.key'+x,x,85,12,7,5,3,{tone:'hi'});
 // The indexing slide has 240 units of travel on a long foreground rail.
 part('index.rail',-125,96,250,3,5,2,{detail:k.lineSvg(k.topTicks(-120,120,8,5,96,7,[1,2.5],P).minor,{tone:'lo'})});
 part('index.stopL',-131,94,4,7,5,5);part('index.stopR',127,94,4,7,5,5);
 part('cursor',-123,96,6,3,7,2,{tone:'hi'});
 A.settle(R,P,{states:[0,.25,.5,.75,1]});
 if(audit)A.auditOrExit(R,P,{argv:['--audit'],frames:[0,.25,.5,.75,1],apart:true});
 const origin=at([screen.x,screen.y,6.02]),x=at([screen.x+screen.w,screen.y,6.02]),y=at([screen.x,screen.y+screen.d,6.02]);
 const matrix=[(x[0]-origin[0])/480,(x[1]-origin[1])/480,(y[0]-origin[0])/260,(y[1]-origin[1])/260,...origin];
 const a=at([0,0,0]),b=at([240,0,0]);
 const art=k.figureSvg({width:640,height:400,label:'Inspection console displaying an illustrative processing sequence.',body:[`<g class="motion-hardware"><path class="iso-halo" d="${k.haloOf(base,-15,10,P)}"/>`,...R.items.sort((a,b)=>a.key-b.key).map(item=>item.svg),'</g>',`<g class="motion-screen" transform="matrix(${matrix.join(' ')})" data-matrix="${matrix.join(' ')}"><svg x="0" y="0" width="480" height="260" viewBox="0 0 480 260" overflow="hidden"><rect width="480" height="260" class="m-screen"/><g data-motion-body>${body}</g></svg></g>`]});
 return {svg:art,offset:[b[0]-a[0],b[1]-a[1]],count:R.solids.length};
}
