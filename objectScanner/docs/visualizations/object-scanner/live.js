const reduced=matchMedia('(prefers-reduced-motion: reduce)');
const panels=[...document.querySelectorAll('.stream')];
const state=panels.map(()=>({position:0,target:0,index:-1,visible:false}));
const refs=panels.map(p=>({p,slider:p.querySelector('.iso-stage'),cursor:p.querySelector('[data-selector]'),parts:[...p.querySelectorAll('[data-component]')],steps:[...p.querySelectorAll('.step')],readout:p.querySelector('[data-readout]'),title:p.querySelector('h3'),description:p.querySelector('.stage-description'),input:p.querySelector('.stage-input'),output:p.querySelector('.stage-output'),count:p.querySelector('.detail-count')}));
let touring=false,frame=0,last=0,elapsed=0,tourStep=0;
const tour=document.getElementById('tour');
const deepRoot=document.getElementById('deep-dive');
const lessonNodes=[...document.querySelectorAll('.lesson')];
const deepLinks=[...document.querySelectorAll('[data-deep-id]')];
let deepIndex=0;
function showDeep(i,j,{url=true}={}){
 deepIndex=i*6+j;
 const player=motionPlayers.get('deep-motion');lessonNodes[deepIndex].querySelector('.lesson-heading').after(player.root);player.setStage(i,j);
 lessonNodes.forEach((node,n)=>{node.hidden=n!==deepIndex;});
 deepLinks.forEach((node,n)=>node.setAttribute('aria-pressed',String(n===deepIndex)));
 document.getElementById('deep-position').textContent=(i===0?'DATASET + TRAINING':'POSE ESTIMATION')+' · 0'+(j+1)+' / 06';
 document.getElementById('deep-prev').disabled=deepIndex===0;
 document.getElementById('deep-next').disabled=deepIndex===11;
 const bottom=document.getElementById('deep-next-bottom');bottom.textContent=deepIndex===11?'Return to the first step →':'Continue to the next step →';
 if(url)history.replaceState(null,'','#'+lessonNodes[deepIndex].id);
}
function stopTour(){touring=false;tourLabel();}
function navigateDeep(n,scroll=false){
 const next=Math.max(0,Math.min(11,n));choose(Math.floor(next/6),next%6);
 if(scroll){document.querySelector('.deep-reader').scrollIntoView({behavior:reduced.matches?'auto':'smooth',block:'start'});lessonNodes[deepIndex].querySelector('h3').focus({preventScroll:true});}
}

function tourLabel(){tour.textContent=reduced.matches?'Next stage':touring?'Pause tour':'Play tour';tour.setAttribute('aria-pressed',String(touring));}
function render(i){
 const s=state[i],r=refs[i],m=stageState(i,s.target),[dx,dy]=offsets[i];
 if(r.cursor)r.cursor.setAttribute('transform',`translate(${dx*s.position/5} ${dy*s.position/5})`);
 if(m.index===s.index)return;
 s.index=m.index;
 motionPlayers.get('motion-'+i).setStage(i,m.index);
 r.readout.textContent=`0${m.index+1} / 06`;r.count.textContent=r.readout.textContent;
 r.title.textContent=m.stage.title;r.description.textContent=m.stage.detail;r.input.textContent=m.stage.input;r.output.textContent=m.stage.output;
 r.slider.setAttribute('aria-valuenow',String(m.index+1));r.slider.setAttribute('aria-valuetext',m.stage.title);
 r.steps.forEach((b,j)=>{b.classList.toggle('selected',j===m.index);b.setAttribute('aria-pressed',String(j===m.index));});
 r.parts.forEach(p=>p.dataset.active=String(p.dataset.component===m.stage.group));
}
function advanceTour(){
 tourStep=(tourStep+1)%12;const i=Math.floor(tourStep/6);state[i].target=tourStep%6;render(i);
 if(!state[i].visible)panels[i].scrollIntoView({behavior:reduced.matches?'auto':'smooth',block:'start'});
}
function tick(now){
 frame=0;const dt=last?Math.min((now-last)/1000,.05):0;last=now;
 if(touring&&!reduced.matches){elapsed+=dt;if(elapsed>=13.5){elapsed=0;advanceTour();}}
 state.forEach((s,i)=>{if(!s.visible)return;s.position=reduced.matches?s.target:s.position+(s.target-s.position)*(1-Math.exp(-dt/.24));if(Math.abs(s.position-s.target)<.001)s.position=s.target;render(i);});
 if(!document.hidden&&state.some(s=>s.visible)&&(touring&&!reduced.matches||state.some(s=>s.visible&&s.position!==s.target)))frame=requestAnimationFrame(tick);
}
function run(){if(!frame&&!document.hidden){last=0;frame=requestAnimationFrame(tick);}}
function choose(i,j){touring=false;tourLabel();state[i].target=Math.max(0,Math.min(5,j));tourStep=i*6+state[i].target;render(i);showDeep(i,state[i].target);document.getElementById('announcement').textContent=streams[i].title+': '+stageState(i,state[i].target).stage.title;run();}
refs.forEach((r,i)=>{
 r.steps.forEach((b,j)=>b.addEventListener('click',()=>choose(i,j)));
 r.slider.addEventListener('keydown',e=>{const direction={ArrowRight:1,ArrowUp:1,ArrowLeft:-1,ArrowDown:-1}[e.key];let next=direction!==undefined?state[i].target+direction:e.key==='Home'?0:e.key==='End'?5:null;if(next===null)return;e.preventDefault();choose(i,next);});

 new IntersectionObserver(es=>{state[i].visible=es.at(-1).isIntersecting;run();},{threshold:.01}).observe(r.p);
 render(i);
});
tour.addEventListener('click',()=>{touring=!touring;elapsed=0;tourLabel();if(reduced.matches&&touring){advanceTour();touring=false;tourLabel();}run();});
document.getElementById('theme').addEventListener('click',e=>{const light=document.body.dataset.theme!=='light';document.body.dataset.theme=light?'light':'dark';e.currentTarget.textContent=light?'Dark mode':'Light mode';});
reduced.addEventListener('change',()=>{if(reduced.matches)touring=false;tourLabel();run();});
document.addEventListener('visibilitychange',()=>{if(document.hidden){cancelAnimationFrame(frame);frame=0;last=0;}else run();});
tourLabel();

deepLinks.forEach((b,n)=>b.addEventListener('click',()=>navigateDeep(n,true)));
document.querySelectorAll('[data-deep-open]').forEach(b=>b.addEventListener('click',()=>{const i=Number(b.dataset.deepOpen);choose(i,state[i].target);deepRoot.scrollIntoView({behavior:reduced.matches?'auto':'smooth'});}));
document.getElementById('deep-prev').addEventListener('click',()=>navigateDeep(deepIndex-1));
document.getElementById('deep-next').addEventListener('click',()=>navigateDeep(deepIndex+1));
document.getElementById('deep-next-bottom').addEventListener('click',()=>navigateDeep((deepIndex+1)%12,true));
deepRoot.addEventListener('pointerenter',stopTour);
deepRoot.addEventListener('focusin',stopTour);
function readDeepHash(){const index=lessonNodes.findIndex(n=>'#'+n.id===location.hash);if(index>=0){navigateDeep(index);requestAnimationFrame(()=>document.querySelector('.deep-reader').scrollIntoView({behavior:'auto'}));}}
window.addEventListener('hashchange',readDeepHash);
readDeepHash();

document.querySelectorAll('.motion-controls').forEach(node=>{node.addEventListener('pointerdown',stopTour);node.addEventListener('focusin',stopTour);});
