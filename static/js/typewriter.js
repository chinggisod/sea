document.addEventListener("DOMContentLoaded", init);

function WaitingFor(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

const KeyEffect = async (wait, content) => {
    var effectTag = document.querySelector('.type-box');
    for (const el of content) {
        await WaitingFor(wait);
        var print;
        for (i = 0; i < el.length; i++) {
            print = el[i];
            effectTag.innerHTML = effectTag.innerHTML + print;
            await WaitingFor(wait);
        }
        await WaitingFor(wait);
        for (j = el.length; j >= 0; j--) {
            var printminus = el.slice(0, j);
            effectTag.innerHTML = printminus;
            await WaitingFor(wait / 4);
        }
    }
    init();
}

function init() {
    const effectTag = document.querySelector('.type-box');
    var wait = effectTag.getAttribute('data-wait');
    var Rawcontent = effectTag.getAttribute('data-content');
    const content = JSON.parse(Rawcontent);
    KeyEffect(wait, content);
}