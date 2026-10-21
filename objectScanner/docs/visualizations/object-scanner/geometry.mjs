import * as k from './vendor/iso-kit.mjs';
import * as A from './vendor/audit.mjs';

export function instrument(kind, audit=false) {
  const base={x:-150,y:-65,w:300,d:130,r:7};
  const P=k.fitProjection([...k.boxCorners(base,-16,0),[-130,-40,90],[130,30,65]],620,370,{pad:36,azimuth:57,elevation:34});
  const R=A.recorder(P);
  const at=p=>k.iso(p,P);
  const L=(d,tone='faint')=>k.lineSvg(d,{tone});
  let serial=0;
  function put(name, plan,z,h,{tone='mid',group='',detail='',move=[0,0,0],ground=false}={}) {
    const shape=A.slab(plan,z,h,{name,ground},6);
    const svg=k.solidSvg(k.slabOf(plan,z,h,P,6,Math.min(.7,h/4)),{tone,lit:name==='selector',inner:detail});
    R.put({name,at:[plan.x+plan.w/2,plan.y+plan.d/2,z+h/2],piece:name==='selector'?'selector':group,move,shapes:[R.solid(shape)],svg:`<g data-component="${group}" ${name==='selector'?'data-selector':''}>${svg}</g>`});
  }
  function box(name,x,y,w,d,z,h,opts={}){ put(name,{x,y,w,d,r:Math.min(1.4,w/5,d/5)},z,h,opts); }
  function round(name,x,y,r,z,h,opts={}){
    const shape=A.cylinder(x,y,r,z,h,{name});
    R.put({name,at:[x,y,z+h/2],piece:opts.group??'',shapes:[R.solid(shape)],svg:`<g data-component="${opts.group??''}">${k.solidSvg(k.cylinder(x,y,r,z,h,P,64,.5),{tone:opts.tone??'mid',inner:opts.detail??''})}</g>`});
  }
  const screws=(plan,z)=>k.dotsSvg(k.corners(plan,3).map(([x,y])=>at([x,y,z])),{size:.52,tone:'hi'});
  const rim=(plan,z)=>L(k.planOutline(k.insetPlan(plan,2),z,P));
  for(const [x,y] of k.corners(base,17)) box('stand.foot'+serial++,x-6,y-6,12,12,-16,6,{tone:'lo',ground:true});
  put('base',base,-10,10,{tone:'lo',detail:rim(base,0)+screws(base,0)});
  for(const y of [-49,45]) box('mounting.rail'+serial++,-137,y,274,4,0,3,{detail:L(k.lineOnTop([-137,y+2],[137,y+2],3,P),'lo')});
  const track={x:-131,y:52,w:262,d:7,r:1};
  const ticks=k.topTicks(-125,125,5,5,52,3,[2,4],P);
  put('selector.track',track,0,3,{tone:'lo',detail:L(ticks.minor,'lo')+L(ticks.major,'mid')});
  for(const x of [-138,134]) box('selector.stop'+serial++,x,50,4,11,0,7,{tone:'lo'});
  box('selector',-125,52,10,7,3,3,{tone:'hi',move:[240,0,0]});
  // RGB-D inspection fixture: mounted camera, real supporting column, specimen platter.
  box('camera.shoe',-133,-40,26,24,0,6,{group:'capture'});
  box('camera.column',-126,-34,10,12,6,54,{group:'capture',detail:L(k.segment([-121,-22,6],[-121,-22,60],P),'lo')});
  box('camera.head',-135,-38,28,20,60,7,{group:'capture'});
  const camera={x:-143,y:-36,w:43,d:22,r:2};
  put('camera.body',camera,67,17,{group:'capture',detail:screws(camera,84)+L(k.sideSeam(camera,78,P,6))});
  box('camera.bezel',-141,-14,39,1.8,69,12,{group:'capture',tone:'hi'});
  for(const x of [-138,-112]) box('camera.depth.window'+x,x,-12.2,7,1,71,8,{group:'capture',tone:'hi'});
  box('camera.rgb.window',-124,-12.2,5,1,72,6,{group:'capture',tone:'hi'});
  round('specimen.pedestal',-91,10,22,0,5,{group:'capture'});
  round('specimen.platter',-91,10,26,5,4,{group:'capture',detail:L(k.ring(-91,10,23,9,P,64),'lo')});
  box('specimen.body',-102,1,22,18,9,19,{group:'capture',tone:'hi'});
  box('specimen.shoulder',-99,3,13,14,28,6,{group:'capture',tone:'hi'});
  // Printed fiducials are ink on their mounted plates.
  for(const [x,y] of [[-108,-7],[-87,-7],[-108,20],[-87,20]]){
    const plan={x,y,w:7,d:7,r:.3};
    const mark=k.dotsSvg([[x+2,y+2],[x+5,y+2],[x+2,y+5],[x+4,y+4]].map(([a,b])=>at([a,b,9.5])),{size:.7,tone:'hi'});
    put('fiducial'+serial++,plan,9,.5,{group:'capture',detail:rim(plan,9.5)+mark});
  }
  if(kind==='dataset'){
    const tray={x:-47,y:-29,w:66,d:55,r:2};
    put('cassette.floor',tray,0,4,{group:'dataset',detail:screws(tray,4)});
    box('cassette.back',-47,-29,66,3,4,29,{group:'dataset'});
    for(const x of [-47,16]) box('cassette.side'+x,x,-26,3,52,4,21,{group:'dataset'});
    for(let i=0;i<3;i++) {
      const card={x:-40,y:-18+i*13,w:52,d:2,r:.4};
      put('image.card'+i,card,4,23-i*3,{group:'dataset',tone:'hi',detail:L(k.sideSeam(card,11+i,P,6),'lo')});
    }
    box('reference.cartridge',-30,29,40,14,0,8,{group:'reference',tone:'hi'});
    box('reference.lid',-28,31,36,10,8,1,{group:'reference',detail:k.dotsSvg(k.dotGrid(-23,33,6,2,4,9,P),{size:.5,tone:'hi'})});
    processor(48,-29,'training');
    for(let i=0;i<2;i++){
      box('model.cartridge'+i,45+i*38,21,30,20,0,7,{group:'training',tone:'hi'});
      box('model.cap'+i,47+i*38,23,26,16,7,1,{group:'training',detail:L(k.planOutline({x:50+i*38,y:26,w:20,d:10,r:.5},8,P),'lo')});
      for(let c=0;c<4;c++) box('model.contact'+serial++,49+i*38+c*5,41,2,3,1,2,{group:'training',tone:'lo'});
    }
  }else{
    const cloud={x:-46,y:-29,w:58,d:39,r:2};
    put('cloud.platen',cloud,0,5,{group:'cloud',detail:rim(cloud,5)+k.dotsSvg(Array.from({length:56},(_,i)=>at([-37+(i%8)*5,-22+Math.floor(i/8)*4,5])),{size:.55,tone:'hi'})});
    box('reference.cartridge',-43,19,48,21,0,7,{group:'refine',tone:'hi'});
    for(let i=0;i<2;i++) box('registration.rail'+i,21,-28+i*35,48,4,0,5,{group:'refine'});
    box('comparison.slide',28,-28,24,39,5,4,{group:'refine',detail:screws({x:28,y:-28,w:24,d:39,r:1},9)});
    box('keypoint.plate',31,-23,18,29,9,2,{group:'pnp',tone:'hi',detail:k.dotsSvg([[34,-19],[46,-19],[34,1],[46,1],[37,-16],[43,-16],[37,-2],[43,-2]].map(([x,y])=>at([x,y,11])),{size:.8,tone:'hi'})});
    processor(20,20,'inference',.58);
    box('output.foot',91,-30,32,28,0,5,{group:'output'});
    box('output.pedestal',99,-24,16,16,5,10,{group:'output'});
    box('output.case',84,-35,46,43,15,27,{group:'output',detail:screws({x:84,y:-35,w:46,d:43,r:1},42)});
    box('output.bezel',88,-31,38,32,42,2,{group:'output',tone:'hi'});
    box('output.glass',91,-28,32,26,44,.4,{group:'output',tone:'lo',detail:[0,1,2].map(i=>k.lineSvg(k.lineOnTop([96,-22+i*6],[115,-22+i*6],44.4,P),{tone:'mid',free:true})).join('')});
    for(const x of [94,111])box('terminal.key'+x,x,17,12,8,0,3,{group:'output'});
  }
  function processor(x,y,group,s=1){
    box('processor.carrier'+group,x,y,72*s,43*s,0,4,{group});
    box('processor.thermal'+group,x+5*s,y+4*s,62*s,35*s,4,5,{group});
    for(let i=0;i<8;i++)box('heatsink.fin'+group+i,x+(7+i*7)*s,y+5*s,2*s,33*s,9,15*s,{group});
  }
  const states=[0,.25,.5,.75,1];
  A.settle(R,P,{states});
  if(audit) A.auditOrExit(R,P,{argv:['--audit'],frames:states,apart:true});
  const svg=k.figureSvg({width:620,height:370,label:kind==='dataset'?'A calibrated capture bench, dataset cassette, reference cartridge and dual-model training processor.':'An RGB-D inspection bench with cloud platen, reference comparator and gated pose terminal.',body:[`<path class="iso-halo" d="${k.haloOf(base,-16,10,P)}"/>`,...R.items.sort((a,b)=>a.key-b.key).map(i=>i.svg)]});
  return {svg,P,count:R.solids.length};
}
