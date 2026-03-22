// ── Chart setup ──────────────────────────────────────────────────────────

const CHART_COLORS = {
  red:    'rgb(255, 99, 132)',
  orange: 'rgb(255, 159, 64)',
  yellow: 'rgb(255, 205, 86)',
  green:  'rgb(75, 192, 192)',
  blue:   'rgb(54, 162, 235)',
  purple: 'rgb(153, 102, 255)',
  grey:   'rgb(201, 203, 207)',
  cayan:  'rgb(75, 192, 192)',
};

let chart_loss = (() => {
  const ctx = document.getElementById('chart_loss');

  const data = {
    datasets: [
      {
        label: 'Train accuracy',
        data: [65, 59, 80, 81, 56, 55, 40],
        fill: false,
        borderColor: CHART_COLORS.red,
        yAxisID: 'y',
        indexAxis: 'x',
      },
      {
        label: 'Test accuracy',
        data: [65, 59, 80, 81, 56, 55, 40],
        fill: false,
        borderColor: CHART_COLORS.blue,
        yAxisID: 'y',
        indexAxis: 'x',
      },
    ],
  };

  const config = {
    type: 'line',
    data,
    options: {
      responsive: true,
      interaction: { mode: 'index', intersect: false },
      stacked: false,
      animation: false,
      plugins: {
        title: { display: true, text: 'Train and Test Accuracy' },
        annotation: {
          annotations: [{
            type: 'line',
            mode: 'horizontal',
            scaleID: 'y',
            value: 10,
            borderColor: CHART_COLORS.cayan,
            borderWidth: 4,
            borderDash: [5, 5],
            label: { enabled: true, content: 'Baseline' },
          }],
        },
      },
      scales: {
        y: { type: 'linear', display: true, position: 'left' },
      },
    },
  };

  return new Chart(ctx, config);
})();

// ── LTTB data decimation ──────────────────────────────────────────────────

function largestTriangleThreeBuckets(data, threshold) {
  const dataLength = data.length;
  if (threshold >= dataLength || threshold === 0) return data;

  const sampled = [];
  let sampledIndex = 0;
  const every = (dataLength - 2) / (threshold - 2);
  let a = 0;
  let maxAreaPoint, maxArea, area, nextA;

  sampled[sampledIndex++] = data[a];

  for (let i = 0; i < threshold - 2; i++) {
    let avgX = 0, avgY = 0;
    let avgRangeStart = Math.floor((i + 1) * every) + 1;
    let avgRangeEnd   = Math.floor((i + 2) * every) + 1;
    avgRangeEnd = avgRangeEnd < dataLength ? avgRangeEnd : dataLength;
    const avgRangeLength = avgRangeEnd - avgRangeStart;

    for (let j = avgRangeStart; j < avgRangeEnd; j++) {
      avgX += data[j].x;
      avgY += data[j].y;
    }
    avgX /= avgRangeLength;
    avgY /= avgRangeLength;

    const rangeOffs = Math.floor((i + 0) * every) + 1;
    const rangeTo   = Math.floor((i + 1) * every) + 1;
    const pointAX = data[a].x;
    const pointAY = data[a].y;
    maxArea = area = -1;

    for (let j = rangeOffs; j < rangeTo; j++) {
      area = Math.abs(
        (pointAX - avgX) * (data[j].y - pointAY) -
        (pointAX - data[j].x) * (avgY - pointAY)
      ) * 0.5;
      if (area > maxArea) {
        maxArea = area;
        maxAreaPoint = data[j];
        nextA = j;
      }
    }

    sampled[sampledIndex++] = maxAreaPoint;
    a = nextA;
  }

  sampled[sampledIndex++] = data[dataLength - 1];
  return sampled;
}

// ── Chart update ──────────────────────────────────────────────────────────

function update_chart(data) {
  const train_accuracy_hist = data['train_accuracy_hist'];
  const test_accuracy_hist  = data['test_accuracy_hist'];
  const percent             = data['percent'];
  const status              = data['status'];
  const baseline_accuracy   = data['baseline_accuracy'];

  const decimated_train = largestTriangleThreeBuckets(
    train_accuracy_hist.map((v, i) => ({ x: i, y: v })), 400
  );
  const decimated_test = largestTriangleThreeBuckets(
    test_accuracy_hist.map((v, i) => ({ x: i, y: v })), 400
  );

  chart_loss.data.labels = decimated_train.map(p => p.x);
  chart_loss.data.datasets[0].data = decimated_train;
  chart_loss.data.datasets[1].data = decimated_test;
  chart_loss.options.plugins.annotation.annotations[0].value = baseline_accuracy;
  chart_loss.update();

  $('#fine_tuning_progress').val(percent);
  $('#fine_tuning_status').text(status);
}

// ── Training buttons ──────────────────────────────────────────────────────

function setTrainingActive(active) {
  document.getElementById('startMusic').disabled    = !!active;
  document.getElementById('startImages').disabled   = !!active;
  document.getElementById('startUniversal').disabled = !!active;
}

// ── Universal training config modal ──────────────────────────────────────

function openUniversalModal() {
  // Reset to defaults
  document.getElementById('train_steps_radio').checked      = true;
  document.getElementById('train_budget_radio').checked     = false;
  document.getElementById('train_steps_input').value        = '5001';
  document.getElementById('train_budget_minutes').value     = '10';
  document.getElementById('train_budget_seconds').value     = '0';
  _syncUniversalModalInputs();
  document.getElementById('train_universal_modal').classList.add('is-active');
}

function closeUniversalModal() {
  document.getElementById('train_universal_modal').classList.remove('is-active');
}

function _syncUniversalModalInputs() {
  const useSteps = document.getElementById('train_steps_radio').checked;
  document.getElementById('train_steps_input').disabled        = !useSteps;
  document.getElementById('train_budget_minutes').disabled     = useSteps;
  document.getElementById('train_budget_seconds').disabled     = useSteps;
}

function confirmUniversalModal() {
  const useSteps = document.getElementById('train_steps_radio').checked;
  const payload = {};

  if (useSteps) {
    const steps = parseInt(document.getElementById('train_steps_input').value, 10);
    if (!steps || steps < 1) {
      document.getElementById('train_steps_input').focus();
      return;
    }
    payload.max_steps = steps;
  } else {
    const minutes = parseInt(document.getElementById('train_budget_minutes').value, 10) || 0;
    const seconds = parseInt(document.getElementById('train_budget_seconds').value, 10) || 0;
    const total_seconds = minutes * 60 + seconds;
    if (total_seconds < 1) {
      document.getElementById('train_budget_minutes').focus();
      return;
    }
    payload.time_budget_seconds = total_seconds;
  }

  closeUniversalModal();
  socket.emit('emit_train_page_start_universal_evaluator_training', payload);
}

// ── Init ──────────────────────────────────────────────────────────────────

$('document').ready(function () {
  socket.on('connect', function () {
    socket.emit('emit_train_page_get_training_status');
  });

  socket.on('emit_train_page_status', function (data) {
    setTrainingActive(data && data.active);
  });

  socket.on('emit_train_page_display_train_data', function (data) {
    update_chart(data);
  });

  socket.on('emit_show_search_status', (status) => {
    console.log('Status:', status);
  });

  // Music / Images: emit immediately (no config needed)
  $('#startMusic').on('click', function () {
    socket.emit('emit_train_page_start_music_evaluator_training');
  });

  $('#startImages').on('click', function () {
    socket.emit('emit_train_page_start_image_evaluator_training');
  });

  // Universal: open config modal first
  $('#startUniversal').on('click', function () {
    openUniversalModal();
  });

  // Modal controls
  document.querySelectorAll('.train-universal-modal-close').forEach(el => {
    el.addEventListener('click', closeUniversalModal);
  });
  document.getElementById('train_universal_confirm').addEventListener('click', confirmUniversalModal);
  document.getElementById('train_universal_cancel').addEventListener('click', closeUniversalModal);

  // Radio buttons toggle input availability
  document.getElementById('train_steps_radio').addEventListener('change', _syncUniversalModalInputs);
  document.getElementById('train_budget_radio').addEventListener('change', _syncUniversalModalInputs);
});
