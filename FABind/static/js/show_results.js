document.addEventListener('DOMContentLoaded', function() {
    // 初始化变量
    let currentPage = 0;
    const rowsPerPage = 10;
    let stage;
  
    // 初始化NGL Viewer
    function initViewer() {
      stage = new NGL.Stage("pdbViewer");
      stage.setParameters({ backgroundColor: "white" }); // 设置背景颜色
    }
  
    // 加载PDB文件到NGL Viewer
    function loadPDB(pdbId) {
      stage.removeAllComponents();
      // stage.loadFile(`static/pdbs/${pdbId}.pdb`, { defaultRepresentation: true });
      stage.loadFile(`static/protein_pdb_files/${pdbId}_protein.pdb`).then(function (o) {
        component = o
        // 设置蛋白质和配体的颜色
        component.addRepresentation("cartoon", { color: "white", opacity: 0.8 });
        o.autoView();
        // o.addRepresentation("cartoon", { color: "white", opacity: 0.2 }); // 蛋白质的颜色
        // o.addRepresentation("surface", { color: "white", opacity: 0.2 });
        // o.addRepresentation("licorice", { color: "green", sele: "ligand" }); // 配体的颜色
        // o.autoView();
      });

      // // 为 Cartoon 按钮添加点击事件处理
      // document.getElementById('btnCartoon').addEventListener('click', function() {
      //   if (component) {
      //     component.removeAllRepresentations(); // 移除当前的表示
      //     component.addRepresentation("cartoon", { color: "white", opacity: 0.8 }); // 添加 cartoon 表示
      //   }
      // });

      // // 为 Surface 按钮添加点击事件处理
      // document.getElementById('btnSurface').addEventListener('click', function() {
      //   if (component) {
      //     component.removeAllRepresentations(); // 移除当前的表示
      //     component.addRepresentation("surface", { color: "white", opacity: 0.1 }); // 添加 surface 表示
      //   }
      // });
      let currentRepresentation = "cartoon"; // 默认为 cartoon

      document.getElementById('btnCartoon').addEventListener('click', function() {
        if (component) {
          component.removeAllRepresentations();
          component.addRepresentation("cartoon", { color: "white", opacity: 0.8 });
          currentRepresentation = "cartoon";
        }
      });

      document.getElementById('btnSurface').addEventListener('click', function() {
        if (component) {
          component.removeAllRepresentations();
          component.addRepresentation("surface", { color: "white", opacity: 0.1 });
          currentRepresentation = "surface";
        }
      });

      document.getElementById('opacitySlider').addEventListener('input', function(event) {
        let opacityValue = event.target.value;
        if (component) {
          component.removeAllRepresentations();
          if (currentRepresentation === "cartoon") {
            component.addRepresentation("cartoon", { color: "white", opacity: opacityValue });
          } else if (currentRepresentation === "surface") {
            component.addRepresentation("surface", { color: "white", opacity: opacityValue });
          }
        }
      });

      stage.loadFile(`static/ligand_sdf_files/${pdbId}.sdf`).then(function (o) {
        o.addRepresentation("ball+stick", { color: "red" });
      });

      stage.loadFile(`static/ligand_sdf_files/${pdbId}_gt.sdf`).then(function (o) {
        o.addRepresentation("ball+stick", { color: "green" });
      });
    }
  
    // 加载和解析TSV文件
    function loadTableData(page) {
      fetch('static/tables/results.csv')
        .then(response => response.text())
        .then(tsvData => {

          const results = Papa.parse(tsvData, {
            header: true,
            skipEmptyLines: true,
          }).data;
          
          const pageData = results.slice(page * rowsPerPage, (page + 1) * rowsPerPage);
          updateTable(pageData);

          if (page === 0 && results.length > 0) {
            loadPDB(results[0].pdb_id); // 加载第一个PDB作为默认显示
          }
        });
    }
  
    // 更新表格
    function updateTable(data) {
      const tableBody = document.getElementById('pdbTable').getElementsByTagName('tbody')[0];
      tableBody.innerHTML = ''; // 清空现有的行
  
      data.forEach(row => {
        const tr = document.createElement('tr');
        console.log(row); // 添加此行来检查解析后的数据
        tr.innerHTML = `<td><a href="https://www.rcsb.org/structure/${row.pdb_id}" target="_blank">${row.pdb_id}</td><td>${row.rmsd}</td>`;
        tableBody.appendChild(tr);
      });
    }
  
    document.getElementById('pdbTable').addEventListener('click', function(event) {
      let target = event.target;
      while (target.tagName !== 'TR') {
        target = target.parentNode;
      }
      const pdbId = target.cells[0].textContent;
    
      // 移除其他所有行的高亮样式
      document.querySelectorAll('#pdbTable tr').forEach(tr => {
        tr.classList.remove('highlighted');
      });
    
      // 为被点击的行添加高亮样式
      target.classList.add('highlighted');
    
      loadPDB(pdbId);
    });

    document.getElementById('opacitySlider').addEventListener('input', function(event) {
      let opacityValue = event.target.value;
      if (component) {
        component.removeAllRepresentations(); // Remove current representation
        component.addRepresentation("cartoon", { color: "white", opacity: opacityValue });
        // Add other representations as needed
      }
    });
  
    // 分页逻辑
    document.querySelector('.pagination-previous').addEventListener('click', function() {
      if (currentPage > 0) {
        currentPage--;
        loadTableData(currentPage);
      }
    });
  
    document.querySelector('.pagination-next').addEventListener('click', function() {
      // 假设有方法来检查是否有更多的页面
      currentPage++;
      loadTableData(currentPage);
    });

  
    // 初始化NGL Viewer并加载第一页数据
    initViewer();
    loadTableData(0);
  });
  